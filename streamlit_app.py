from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
import uuid
import datetime as dt
import time
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pytz
import requests

st.set_page_config(
    page_title="Stat",
    page_icon="😎",
)

st.write(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)

DEFAULTSIZE: int = 500


class BaseStatsCalculator(ABC):
    @abstractmethod
    def calculate_stats(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        pass


class BaseDataTransformer(ABC):
    @abstractmethod
    def validate_uuid(self, uid: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def get_top_posts(
        df: pd.DataFrame, n: int, nsfw: Optional[bool] = None
    ) -> pd.DataFrame:
        pass


class APIClient:
    def __init__(self, user_id: str, data_transformer: "BaseDataTransformer"):
        self._API_URL = (
            "https://api.yodayo.com/v1/users/{user_id}/posts?include_nsfw=true"
        )
        self._user_id = user_id
        self.data_transformer = data_transformer
        self.session = requests.Session()

    def _format_api_url(self, user_id: str) -> str:
        return self._API_URL.format(user_id=user_id)

    def check_user_exists(self, uid: str) -> bool:
        try:
            response = self.session.get(self._format_api_url(uid), timeout=10)
            response.raise_for_status()
            return True
        except requests.HTTPError:
            return False

    @st.cache_data(ttl=7200)
    def get_posts(
        _self, user_id: str, limit: int = DEFAULTSIZE, offset: int = 0
    ) -> List[Dict]:
        params = {"limit": limit, "offset": offset}
        headers = {"Accept-Encoding": "gzip, deflate"}
        posts = []
        while True:
            try:
                url = _self._format_api_url(user_id)
                response = _self.session.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                batch_posts = data.get("posts", []) if isinstance(data, dict) else data
                if not batch_posts:
                    break
                current_length = len(posts)
                posts.extend(_self.data_transformer.standardize_posts(batch_posts))
                if len(batch_posts) < limit:
                    break
                params["offset"] += len(batch_posts)
                if len(posts) == current_length:
                    break
            except requests.exceptions.RequestException:
                break
        return posts


class YDStats:
    def __init__(
        self,
        user_id: str,
        data_transformer: BaseDataTransformer,
        stats_calculator: BaseStatsCalculator,
        api_client: APIClient,
    ):
        self._user_id = user_id
        self.data_transformer = data_transformer
        self.stats_calc = stats_calculator
        self.api_client = api_client
        self.metrics_calculator = MetricsCalculator()
        self.posts = None
        self.df = None
        self.visualizer = Visualizer(self.clean_names)

    def validate_uuid(self):
        return self.data_transformer.validate_uuid(self._user_id)

    def fetch_data(self, posts):
        self.posts = posts
        is_valid_uuid = self.validate_uuid()
        user_exists = (
            self.api_client.check_user_exists(self._user_id) if is_valid_uuid else False
        )
        if not is_valid_uuid:
            st.warning("Please enter a valid UUID")
        elif not user_exists:
            st.error("User ID not found")
        else:
            if len(self.posts) == 0:
                st.warning("This user has no posts")
            else:
                self.show_metrics()
                self.process_user_data()

    def show_metrics(self):
        stats, nsfw_df = self.stats_calc.calculate_stats(pd.DataFrame(self.posts))

    def clean_names(self, names: pd.Series) -> pd.Series:
        return (
            names.str.replace("v1", "", regex=False)
            .str.strip()
            .str.replace("v2 v2", "v2", regex=False)
        )

    def get_model_info(self) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        if not self.posts:
            st.warning("No posts fetched. Please fetch data first.")
            return None, None, None
        df = pd.DataFrame(self.posts)
        cleaned_model_names = self.clean_names(
            df["model_display_names"].explode().dropna()
        )
        sampling_method_names = df["sampling_method_display_names"].explode().dropna()
        cfg_scales = df["cfg_scales"].explode().dropna()
        if (
            not cleaned_model_names.any()
            and not sampling_method_names.any()
            and not cfg_scales.any()
        ):
            return None, None, None
        with pd.option_context("mode.chained_assignment", None):
            most_popular_model = (
                cleaned_model_names.value_counts().idxmax()
                if cleaned_model_names.any()
                else None
            )
            most_popular_sampling_method = (
                sampling_method_names.value_counts().idxmax()
                if sampling_method_names.any()
                else None
            )
            avg_cfg_scale = cfg_scales.mean() if cfg_scales.any() else None
        return most_popular_model, most_popular_sampling_method, avg_cfg_scale

    def add_url_column(self, df):
        new_df = df[["title", "likes", "id"]].copy()
        new_df = new_df.convert_dtypes()
        new_df["url"] = "https://yodayo.com/posts/" + new_df["id"]
        return new_df

    def render_dataframe(self, df, header_text, display_text_pattern):
        column_config = {
            "url": st.column_config.LinkColumn(
                "Link", display_text=display_text_pattern
            )
        }
        st.header(header_text)
        st.dataframe(
            df[["title", "likes", "url"]],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    def show_posts_below(self, df: pd.DataFrame, threshold: int) -> None:
        filtered_df = df[df["likes"] < threshold]
        count = len(filtered_df)
        columns = ["title", "likes", "url"]
        filtered_df = self.add_url_column(filtered_df)
        st.dataframe(
            filtered_df[columns],
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Link", display_text="(.*)")
            },
            hide_index=True,
        )

    def process_user_data(self):
        if self.posts is None:
            st.warning("No posts fetched. Please fetch data first.")
            return
        df = pd.DataFrame(self.posts)
        sfw_df = df[~df["nsfw"]]
        stats, nsfw_df = self.stats_calc.calculate_stats(df)
        num_posts = st.number_input(
            "Number of top posts to show", min_value=1, max_value=500, value=10
        )
        include_nsfw = st.checkbox(f"Include NSFW🌶️ posts in TOP-{num_posts}")
        if include_nsfw:
            top_posts_df = nsfw_df.head(num_posts)
        else:
            top_posts_df = df[~df["nsfw"]].head(num_posts)
        nsfw = include_nsfw
        top_posts = self.data_transformer.get_top_posts(df, num_posts, nsfw=nsfw)
        metrics = MetricsCalculator.get_orientation_metrics(df)
        most_posts_date = self.data_transformer.get_date_with_most_posts(df)
        mean_time = self.data_transformer.get_mean_time_between_posts(df)
        first_post_date = df["timestamp"].min().floor("D")
        longest_pause = self.data_transformer.get_longest_pause_between_posts(df)
        landscape_posts = self.data_transformer.get_landscape_posts(df)
        square_posts = self.data_transformer.get_square_posts(df)
        (
            model_display_name,
            sampling_method_display_name,
            avg_cfg_scale,
        ) = self.get_model_info()
        row1, row2, row3 = st.columns(3)
        new_row1, new_row2, new_row3 = st.columns(3)
        with row1:
            st.metric("Total posts", stats["num_posts"])
            st.metric(
                "№ Portrait posts:", metrics["orientation_counts"].get("portrait", 0)
            )
            st.metric(
                "№ Landscape posts:", metrics["orientation_counts"].get("landscape", 0)
            )
            st.metric("№ Square posts:", metrics["orientation_counts"].get("square", 0))
            st.metric(
                "NSFW posts🌶️",
                f"{stats['nsfw_metrics']['nsfw_count']} ({stats['nsfw_metrics']['nsfw_pct']}%)",
            )
            p50 = round(np.percentile(df["likes"], 50), 0)
            st.metric(
                "50% Percentile",
                f"{p50:,} likes",
                f"{(df['likes'] > p50).sum():,} posts",
            )
            st.metric("Date with most posts", most_posts_date)
        with row2:
            st.metric("Total likes", stats["total_likes"])
            st.metric(
                "% of Portrait",
                f"{metrics['orientation_distribution'].get('portrait', 0):.2f}%",
            )
            st.metric(
                "% of Landscape",
                f"{metrics['orientation_distribution'].get('landscape', 0):.2f}%",
            )
            st.metric(
                "% of Square",
                f"{metrics['orientation_distribution'].get('square', 0):.2f}%",
            )
            st.metric("NSFW likes🌶️", stats["nsfw_metrics"]["nsfw_likes"])
            p75 = round(np.percentile(df["likes"], 75), 0)
            st.metric(
                "75% Percentile ",
                f"{p75:,} likes",
                f"{(df['likes'] > p75).sum():,} posts",
            )
            st.metric("Mean time between posts", mean_time)
        with row3:
            st.metric("Avg likes/post", stats["avg_likes"])
            st.metric(
                "Avg Likes for Portrait",
                f"{metrics['avg_likes_by_orientation'].get('portrait', 0):.2f}",
            )
            st.metric(
                "Avg Likes for Landscape",
                f"{metrics['avg_likes_by_orientation'].get('landscape', 0):.2f}",
            )
            st.metric(
                "Avg Likes for Square",
                f"{metrics['avg_likes_by_orientation'].get('square', 0):.2f}",
            )
            st.metric("Avg posts/day", round(stats["avg_posts_per_day"], 2))
            p90 = round(np.percentile(df["likes"], 90), 0)
            st.metric(
                "90% Percentile",
                f"{p90:,} likes",
                f"{(df['likes'] > p90).sum():,} posts",
            )
            st.metric("First Post Date", first_post_date.strftime("%b %d, %Y"))
        with new_row1:
            st.metric("Most Used Model:", model_display_name)
        with new_row2:
            st.metric("Most Used Sampling Method:", sampling_method_display_name)
        with new_row3:
            st.metric("Longest pause between posts:", longest_pause)
        # Top charts and data
        column_config = {
            "url": st.column_config.LinkColumn("Link", display_text="(.*)")
        }
        top_posts = self.data_transformer.get_top_posts(df[~df["nsfw"]], num_posts)
        top_posts = self.add_url_column(top_posts)
        nsfw_top_posts = self.data_transformer.get_top_posts(nsfw_df, num_posts, nsfw=True)
        nsfw_top_posts = self.add_url_column(nsfw_top_posts)
        if include_nsfw:
            combined_top_posts = pd.concat([top_posts, nsfw_top_posts], ignore_index=True)
            combined_top_posts = combined_top_posts.sort_values(by="likes", ascending=False).head(num_posts)
            combined_top_posts = self.add_url_column(combined_top_posts)
            self.render_dataframe(combined_top_posts, f"TOP-{num_posts} posts", "(.*)")
        else:
            self.render_dataframe(top_posts, f"TOP-{num_posts} posts", "(.*)")

        self.render_dataframe(nsfw_top_posts, f"TOP-{num_posts} NSFW🌶️ posts", "(.*)")
        oldest_posts = self.data_transformer.get_extreme_posts(
            df, num_posts, sort="oldest"
        )
        oldest_posts = self.add_url_column(oldest_posts)
        self.render_dataframe(oldest_posts, f"Oldest {num_posts} Posts🕸", "(.*)")
        newest_posts = self.data_transformer.get_extreme_posts(
            df, num_posts, sort="newest"
        )
        newest_posts = self.add_url_column(newest_posts)
        self.render_dataframe(newest_posts, f"Latest {num_posts} Posts🆕", "(.*)")
        landscape_posts = self.add_url_column(landscape_posts)
        self.render_dataframe(
            landscape_posts, "Landscape Posts (width > height)", "(.*)"
        )
        square_posts = self.add_url_column(square_posts)
        self.render_dataframe(square_posts, "Square Posts (width = height)", "(.*)")
        threshold = st.number_input("Max likes", min_value=0, value=10)
        header_text = f"Posts with less than {threshold} likes"
        st.header(header_text)
        self.show_posts_below(df, threshold)
        st.metric(f"Posts below {threshold} likes:", len(df[df["likes"] < threshold]))
        # Plots
        st.pyplot(Visualizer.plot_likes_per_day(df))
        st.pyplot(Visualizer.plot_likes_per_hour(df))
        st.altair_chart(Visualizer.plot_likes_per_week(df))
        model_distribution_plot = self.visualizer.plot_distribution(df, "model")
        sampling_method_distribution_plot = self.visualizer.plot_distribution(
            df, "sampling_method"
        )
        # Raw data. likes text
        with st.expander("See raw data likes"):
            st.write(self.data_transformer.get_weekly_likes(df))


class StatsCalculator(BaseStatsCalculator):
    def calculate_stats(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        num_posts = df.shape[0]
        total_likes = df["likes"].sum()
        avg_likes = round(df["likes"].mean(), 2)
        median_likes = df["likes"].median()
        start_date = df.index.min().floor("D")
        end_date = df.index.max().ceil("D")
        num_days = (end_date - start_date).days + 1
        avg_posts_per_day = num_posts / num_days
        nsfw_df = df[df["nsfw"]]
        nsfw_count = nsfw_df.shape[0]
        nsfw_likes = nsfw_df["likes"].sum()
        nsfw_pct = round(nsfw_count / num_posts * 100, 2) if num_posts > 0 else 0
        stats = {
            "num_posts": num_posts,
            "total_likes": total_likes,
            "avg_likes": avg_likes,
            "median_likes": median_likes,
            "avg_posts_per_day": avg_posts_per_day,
            "nsfw_metrics": {
                "nsfw_count": nsfw_count,
                "nsfw_pct": nsfw_pct,
                "nsfw_likes": nsfw_likes,
            },
        }
        return stats, nsfw_df


class DataTransformer(BaseDataTransformer):
    def __init__(self, stats_calc: "BaseStatsCalculator"):
        self.stats_calc = stats_calc

    def calculate_stats(self, df):
        return self.stats_calc.calculate_stats(df)

    def validate_uuid(self, uid) -> bool:
        try:
            uuid.UUID(uid)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_top_posts(
        df: pd.DataFrame, n: int, nsfw: Optional[bool] = None
    ) -> pd.DataFrame:
        if nsfw is None:
            nsfw = False
        sfw_df = df[~df["nsfw"]]
        nsfw_df = df[df["nsfw"]]
        if nsfw:
            top_posts = pd.concat([nsfw_df, sfw_df]).nlargest(n, "likes")
        else:
            top_posts = sfw_df.nlargest(n, "likes")
        return top_posts[["title", "likes", "id"]]

    @staticmethod
    def get_nsfw_metrics(df: pd.DataFrame) -> dict:
        nsfw_df = df[df["nsfw"]]
        return {
            "nsfw_count": len(nsfw_df),
            "nsfw_pct": round(len(nsfw_df) / len(df) * 100, 2) if len(df) > 0 else 0,
            "nsfw_likes": nsfw_df["likes"].sum(),
        }

    def standardize_posts(self, data: List[Dict]) -> List[Dict]:
        timestamps = pd.to_datetime(
            [post.get("created_at") for post in data], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        standardized_data = []
        for post, timestamp in zip(data, timestamps):
            photo_media = post.get("photo_media", [])

            if photo_media is not None:
                width = (
                    photo_media[0].get("width") if photo_media else post.get("width")
                )
                height = (
                    photo_media[0].get("height") if photo_media else post.get("height")
                )
                model_display_names = [
                    media.get("text_to_image", {}).get("model_display_name")
                    for media in photo_media
                ]
                sampling_method_display_names = [
                    media.get("text_to_image", {}).get("sampling_method_display_name")
                    for media in photo_media
                ]
                cfg_scales = [
                    media.get("text_to_image", {}).get("cfg_scale")
                    for media in photo_media
                ]
            else:
                width = post.get("width")
                height = post.get("height")
                model_display_names = []
                sampling_method_display_names = []
                cfg_scales = []
            standardized_post = {
                "id": post.get("uuid"),
                "likes": post.get("likes"),
                "timestamp": timestamp,
                "title": post.get("title"),
                "nsfw": post.get("nsfw"),
                "width": width,
                "height": height,
                "model_display_names": model_display_names,
                "sampling_method_display_names": sampling_method_display_names,
                "cfg_scales": cfg_scales,
            }
            standardized_data.append(standardized_post)
        return standardized_data

    @staticmethod
    def get_longest_pause_between_posts(df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        df = df.sort_values("timestamp", ascending=True)
        time_diffs = df["timestamp"].diff().dropna()
        tz = df["timestamp"].dt.tz
        if tz is not None:
            last_post = pd.Timestamp.now(tz=tz)
        else:
            last_post = pd.Timestamp(dt.datetime.now())
        time_since_last = last_post - df["timestamp"].iloc[-1]
        longest_pause = max(time_diffs.max(), time_since_last)
        days = longest_pause.components.days
        hours = longest_pause.components.hours
        return f"{days} d, {hours} h"

    @staticmethod
    def get_weekly_likes(df: pd.DataFrame) -> pd.Series:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        weekly_likes = df.resample("W-Mon").sum()["likes"]
        last_week_end = weekly_likes.index[-1] + pd.DateOffset(days=6)
        last_week_end = last_week_end.tz_localize(None)
        if last_week_end > pd.to_datetime("today"):
            weekly_likes = weekly_likes.iloc[:-1]
        end_of_week = weekly_likes.index + pd.DateOffset(days=6)
        formatted_index = (
            weekly_likes.index.strftime("%b %d %Y")
            + " - "
            + end_of_week.strftime("%b %d %Y")
        )
        weekly_likes.index = formatted_index
        return weekly_likes

    @staticmethod
    def get_landscape_posts(df):
        return df[df["width"] > df["height"]]

    @staticmethod
    def get_square_posts(df):
        return df[df["width"] == df["height"]]

    @staticmethod
    def get_date_with_most_posts(df) -> str:
        posts_per_date = df.groupby(df["timestamp"].dt.date)["id"].count()
        most_posts_date = posts_per_date.idxmax()
        most_posts_count = posts_per_date.max()
        return f"{most_posts_date.strftime('%d.%m.%y')} ({most_posts_count})"

    @staticmethod
    def get_mean_time_between_posts(df: pd.DataFrame) -> Optional[str]:
        if len(df) < 2:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        time_diff_seconds = df.index.to_series().diff().dt.total_seconds().dropna()
        mean_elapsed_seconds = time_diff_seconds.mean()
        mean_elapsed_minutes = round(abs(mean_elapsed_seconds) / 60)

        return f"{mean_elapsed_minutes} minutes"

    def get_extreme_posts(
        self, df: pd.DataFrame, n: int, sort: str = "oldest"
    ) -> pd.DataFrame:
        columns = ["title", "likes", "id", "timestamp"]
        sorted_df = df.sort_values("timestamp", ascending=(sort == "oldest"))
        return sorted_df.head(n)[columns]


class Visualizer:
    def __init__(self, clean_names_func):
        self.clean_names = clean_names_func

    @staticmethod
    def setup_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="both", color="gray", linestyle="dashed")
        plt.tight_layout()

    @staticmethod
    def plot_likes_per_day(df: pd.DataFrame) -> plt.Figure:
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        daily_likes = (
            df.groupby(df["timestamp"].dt.day_name())["likes"].sum().reindex(day_order)
        )
        fig, ax = plt.subplots()
        sns.barplot(
            x=daily_likes.index,
            y=daily_likes,
            ax=ax,
            hue=daily_likes.index,
            palette="viridis",
            legend=False,
        )
        Visualizer.setup_plot(ax, "Likes per Day of Week", "Day of Week", "Total Likes")
        plt.show()

        return fig

    @staticmethod
    def plot_likes_per_hour(df: pd.DataFrame) -> plt.Figure:
        timezones: Dict[str, pytz.timezone] = {
            "UTC": pytz.utc,
            "US/East": pytz.timezone("US/Eastern"),
            "US/Pacific": pytz.timezone("US/Pacific"),
            "Brazil/East": pytz.timezone("Brazil/East"),
            "Europe/Berlin,Warsaw,Budapest": pytz.timezone("Europe/Berlin"),
            "Europe/Kiev": pytz.timezone("Europe/Kiev"),
            "Asia/Tokyo": pytz.timezone("Asia/Tokyo"),
        }
        tz = st.selectbox("Select Timezone", list(timezones.keys()))
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezones[tz])
        hourly_likes = df.groupby(df["timestamp"].dt.hour)["likes"].sum()
        fig, ax = plt.subplots()
        sns.barplot(
            x=hourly_likes.index,
            y=hourly_likes,
            ax=ax,
            hue=hourly_likes.index,
            palette="viridis",
            legend=False,
        )
        Visualizer.setup_plot(ax, "Likes per Hour", "Hour", "Total Likes")
        plt.show()
        return fig

    @staticmethod
    def plot_likes_per_week(df: pd.DataFrame) -> alt.Chart:
        weekly_likes = df.set_index("timestamp").resample("W-MON").sum()["likes"]
        weekly_likes = weekly_likes.reset_index()
        weekly_likes.columns = ["week", "likes"]
        min_timestamp = weekly_likes["week"].min()
        max_timestamp = weekly_likes["week"].max()
        xmin = min_timestamp.to_pydatetime()
        xmax = max_timestamp.to_pydatetime() + timedelta(days=6)
        xmin, xmax = st.slider(
            "Select Date Range", min_value=xmin, max_value=xmax, value=(xmin, xmax)
        )
        mask = (weekly_likes["week"] >= pd.to_datetime(xmin)) & (
            weekly_likes["week"] <= pd.to_datetime(xmax)
        )
        filtered_df = weekly_likes.loc[mask]
        current_week = datetime.now().isocalendar()[1]
        filtered_df = filtered_df[
            filtered_df["week"].dt.isocalendar().week != current_week + 1
        ]
        filtered_df["week"] = pd.to_datetime(filtered_df["week"])
        first_week = filtered_df[filtered_df["week"] == filtered_df["week"].min()]
        last_week = filtered_df[filtered_df["week"] == filtered_df["week"].max()]
        middle_weeks = filtered_df[
            (filtered_df["week"] != filtered_df["week"].min())
            & (filtered_df["week"] != filtered_df["week"].max())
        ]
        filtered_chart = alt.Chart(filtered_df).interactive()
        if not first_week.empty:
            filtered_chart = filtered_chart.mark_point(color="red").encode(
                x="week", y="likes", tooltip=["week", "likes"]
            )
        if not last_week.empty:
            filtered_chart = filtered_chart.mark_point(color="red").encode(
                x="week", y="likes", tooltip=["week", "likes"]
            )
        filtered_chart = filtered_chart.mark_line().encode(
            x="week", y="likes", tooltip=["week", "likes"]
        )
        return filtered_chart

    def plot_distribution(
        self, df: pd.DataFrame, plot_type: str, min_appearances: int = 10
    ):
        if df.empty:
            st.warning("DataFrame is empty. Please check your data.")
            return
        if plot_type == "model":
            column_name = "model_display_names"
            x_label = "Model Name"
            title = "Model Distribution"
        elif plot_type == "sampling_method":
            column_name = "sampling_method_display_names"
            x_label = "Sampling Method"
            title = "Sampling Method Distribution"
        else:
            st.warning(f"Invalid plot type: {plot_type}")
            return
        flattened_names = df[column_name].explode().dropna()
        cleaned_names = self.clean_names(flattened_names)
        name_counts = cleaned_names.value_counts()
        if name_counts.empty:
            st.warning(f"No data to visualize for {plot_type} distribution.")
            return
        name_counts = name_counts[name_counts >= min_appearances].sort_values(
            ascending=False
        )
        if not name_counts.empty:
            fig, ax = plt.subplots(figsize=(12, len(name_counts) * 0.5))
            name_counts.plot(kind="barh", ax=ax, color="skyblue")

            for index, value in enumerate(name_counts):
                ax.text(
                    value + 0.1,
                    index,
                    f" {value}",
                    va="center",
                    color="black",
                    fontweight="bold",
                )
            ax.set_xlabel("Count")
            ax.set_ylabel(x_label)
            ax.set_title(f"{title} (>={min_appearances} uses)")
            st.pyplot(fig)
        else:
            st.warning(f"No data to visualize for {plot_type} distribution.")


class MetricsCalculator:
    @staticmethod
    def get_orientation_metrics(posts: List[Dict]) -> Dict:
        df = pd.DataFrame(posts)
        df["orientation"] = np.where(
            df["width"] == df["height"],
            "square",
            np.where(df["width"] > df["height"], "landscape", "portrait"),
        )
        orientation_distribution = df["orientation"].value_counts(normalize=True) * 100
        orientation_counts = dict(df["orientation"].value_counts().items())
        avg_likes_by_orientation = {
            orient: stats["likes"].mean() for orient, stats in df.groupby("orientation")
        }
        return {
            "orientation_distribution": orientation_distribution.to_dict(),
            "orientation_counts": orientation_counts,
            "avg_likes_by_orientation": avg_likes_by_orientation,
        }


def main():
    st.title("YD Stats")
    user_id = st.query_params.get("user_id", "")
    user_id = st.text_input("Enter your user ID", value=user_id, key="user_id_input")
    if user_id:
        start_time = time.perf_counter()
        stats_calc = StatsCalculator()
        transformer = DataTransformer(stats_calc)
        api_client = APIClient(user_id, transformer)
        posts = api_client.get_posts(user_id)
        stats = YDStats(user_id, transformer, stats_calc, api_client)
        stats.fetch_data(posts)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        st.write(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
