import ephem
from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
import uuid
import datetime as dt
import time
import streamlit as st
import numpy as np
import pandas as pd
import pytz
import requests
import altair as alt

st.set_page_config(
    page_title="Stat",
    page_icon="😎",
)

st.write(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)

DEFAULTSIZE: int = 500
SPECIAL_USER_IDS = [
    "0b56f7c9-b80a-4b3e-9a9f-36b038898b1b",
    "24646f3c-0deb-4446-a91d-d6a283c60e42",
]
SPECIAL_SIZE: int = 20


def human_moon(observer):
    target_date_utc = observer.date
    target_date_local = ephem.localtime(target_date_utc).date()
    next_full = ephem.localtime(ephem.next_full_moon(target_date_utc)).date()
    next_new = ephem.localtime(ephem.next_new_moon(target_date_utc)).date()
    next_last_quarter = ephem.localtime(
        ephem.next_last_quarter_moon(target_date_utc)
    ).date()
    next_first_quarter = ephem.localtime(
        ephem.next_first_quarter_moon(target_date_utc)
    ).date()
    previous_full = ephem.localtime(ephem.previous_full_moon(target_date_utc)).date()
    previous_new = ephem.localtime(ephem.previous_new_moon(target_date_utc)).date()
    previous_last_quarter = ephem.localtime(
        ephem.previous_last_quarter_moon(target_date_utc)
    ).date()
    previous_first_quarter = ephem.localtime(
        ephem.previous_first_quarter_moon(target_date_utc)
    ).date()
    if target_date_local in (next_full, previous_full):
        return "Full"
    elif target_date_local in (next_new, previous_new):
        return "New"
    elif target_date_local in (next_first_quarter, previous_first_quarter):
        return "First Quarter"
    elif target_date_local in (next_last_quarter, previous_last_quarter):
        return "Last Full Quarter"
    elif previous_new < next_first_quarter < next_full < next_last_quarter < next_new:
        return "Waxing Crescent"
    elif (
        previous_first_quarter
        < next_full
        < next_last_quarter
        < next_new
        < next_first_quarter
    ):
        return "Waxing Gibbous"
    elif previous_full < next_last_quarter < next_new < next_first_quarter < next_full:
        return "Waning Gibbous"
    elif (
        previous_last_quarter
        < next_new
        < next_first_quarter
        < next_full
        < next_last_quarter
    ):
        return "Waning Crescent"


def get_moon_phase(date):
    if isinstance(date, str):
        return date
    else:
        moon_phase_emojis = {
            "Full": "🌕 Full Moon",
            "New": "🌑 New Moon",
            "First Quarter": "🌓 First Quarter",
            "Last Full Quarter": "🌗 Last Quarter",
            "Waxing Crescent": "🌒 Waxing Crescent",
            "Waxing Gibbous": "🌔 Waxing Gibbous",
            "Waning Gibbous": "🌖 Waning Gibbous",
            "Waning Crescent": "🌘 Waning Crescent",
        }
        observer = ephem.Observer()
        observer.date = ephem.Date(date)
        moon_phase = human_moon(observer)
        return moon_phase_emojis[moon_phase]


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
            response = self.session.get(self._format_api_url(uid))
            response.raise_for_status()
            return True
        except requests.HTTPError:
            return False

    @st.cache_data(ttl=86400)
    def get_posts(
        _self, user_id: str, limit: int = None, offset: int = 0
    ) -> List[Dict]:
        if limit is None:
            limit = SPECIAL_SIZE if user_id in SPECIAL_USER_IDS else DEFAULTSIZE

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

    def get_model_info(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
        if not self.posts:
            st.warning("No posts fetched. Please fetch data first.")
            return None, None, None, None
        df = pd.DataFrame(self.posts)
        cleaned_model_names = self.clean_names(
            df["model_display_names"].explode().dropna()
        )
        sampling_method_names = df["sampling_method_display_names"].explode().dropna()
        cfg_scales = df["cfg_scales"].explode().dropna()
        sampling_steps = df["sampling_steps"].explode().dropna()
        if (
            not cleaned_model_names.any()
            and not sampling_method_names.any()
            and not cfg_scales.any()
            and not sampling_steps.any()
        ):
            return None, None, None, None
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
            avg_sampling_steps = sampling_steps.mean() if sampling_steps.any() else None
        return (
            most_popular_model,
            most_popular_sampling_method,
            avg_cfg_scale,
            avg_sampling_steps,
        )

    def add_url_column(self, df):
        new_df = df[["title", "likes", "id"]].copy()
        new_df = new_df.convert_dtypes()
        new_df["url"] = "https://yodayo.com/posts/" + new_df["id"]
        return new_df

    def render_dataframe(self, df, header_text, display_text_pattern):
        column_config = {
            "url": st.column_config.LinkColumn(
                "Link", display_text=display_text_pattern
            ),
            "title": st.column_config.TextColumn(
                "Title",
            ),
            "likes": st.column_config.TextColumn(
                "Likes",
            ),
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
        filtered_df = self.add_url_column(filtered_df)
        count = len(filtered_df)
        header_text = f"Posts with less than {threshold} likes ({count})"
        self.render_dataframe(filtered_df, header_text, "(.*)")

    def plot_moon_phase_frequency(self, df):
        moon_phase_counts = df["moon_phase"].value_counts().reset_index()
        moon_phase_counts.columns = ["moon_phase", "count"]
        chart = (
            alt.Chart(moon_phase_counts)
            .mark_bar()
            .encode(
                x=alt.X("count", title="Number of Posts"),
                y=alt.Y(
                    "moon_phase",
                    title="Moon Phase",
                    sort=None,
                    axis=alt.Axis(labelFontSize=12),
                ),
                tooltip=["moon_phase", "count"],
            )
            .properties(title="Post Frequency by Moon Phase")
        )
        st.altair_chart(chart, use_container_width=True)

    @st.experimental_fragment
    def process_user_data(self):
        if self.posts is None:
            st.warning("No posts fetched. Please fetch data first.")
            return
        df = pd.DataFrame(self.posts)
        sfw_df = df[~df["nsfw"]]
        stats, nsfw_df = self.stats_calc.calculate_stats(df)
        num_posts = st.number_input(
            "Number of top posts to show",
            min_value=1,
            max_value=2000,
            value=10,
            help="Number is adjustable and will affect dataframes. (Min 1, Max 2000)",
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
            avg_sampling_steps,
        ) = self.get_model_info()
        row1, row2, row3 = st.columns(3)
        row4, row5, row6 = st.columns(3)
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
                help="This metric shows the median number of likes for all posts, which means that half of the posts have less likes than this value.",
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
                help="Upper third quartile. 75% of posts are below this value",
            )
            st.metric(
                "Mean time between posts",
                mean_time,
                help="Shows the average time between your posts.",
            )
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

            if stats["nsfw_metrics"]["nsfw_count"] > 0:
                avg_nsfw_likes = (
                    stats["nsfw_metrics"]["nsfw_likes"]
                    / stats["nsfw_metrics"]["nsfw_count"]
                )
                st.metric("Average NSFW likes🌶️", f"{avg_nsfw_likes:.2f}")
            else:
                st.metric("Average NSFW likes🌶️", 0)
            p90 = round(np.percentile(df["likes"], 90), 0)
            st.metric(
                "90% Percentile",
                f"{p90:,} likes",
                f"{(df['likes'] > p90).sum():,} posts",
                help="Top-10% of posts value and count",
            )
            st.metric("First Post Date", first_post_date.strftime("%b %d, %Y"))
        with row4:
            st.metric(
                "Most Used Model:",
                model_display_name,
                help="This takes into account all album posts and not only first one. Works only for images with shown prompt.",
            )
            st.metric("Most Used Sampling Method:", sampling_method_display_name)
        with row5:
            st.metric("Avg posts/day", round(stats["avg_posts_per_day"], 2))
            st.metric(
                "Longest pause between posts:",
                longest_pause,
                help="Identifies the maximum time gap between consecutive posts.",
            )
        with row6:
            st.metric("Avg sampling steps", round(avg_sampling_steps or 0, 2))
            st.metric("Avg cfg scale used", round(avg_cfg_scale or 0, 2))
        # Top charts and data
        top_posts = self.data_transformer.get_top_posts(df[~df["nsfw"]], num_posts)
        top_posts = self.add_url_column(top_posts)
        nsfw_top_posts = self.data_transformer.get_top_posts(
            nsfw_df, num_posts, nsfw=True
        )
        nsfw_top_posts = self.add_url_column(nsfw_top_posts)
        if include_nsfw:
            combined_top_posts = pd.concat(
                [top_posts, nsfw_top_posts], ignore_index=True
            )
            combined_top_posts = combined_top_posts.sort_values(
                by="likes", ascending=False
            ).head(num_posts)
            combined_top_posts = self.add_url_column(combined_top_posts)
            self.render_dataframe(combined_top_posts, f"TOP-{num_posts} posts⭐", "(.*)")
        else:
            self.render_dataframe(top_posts, f"TOP-{num_posts} posts⭐", "(.*)")
        if not nsfw_top_posts.empty:
            self.render_dataframe(
                nsfw_top_posts, f"TOP-{num_posts} NSFW🌶️ posts", "(.*)"
            )
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
        if not landscape_posts.empty:
            landscape_posts = self.add_url_column(landscape_posts)
            self.render_dataframe(
                landscape_posts, "Landscape Posts (width > height)", "(.*)"
            )
        if not square_posts.empty:
            square_posts = self.add_url_column(square_posts)
            self.render_dataframe(square_posts, "Square Posts (width = height)", "(.*)")
        threshold = st.number_input("Max likes", min_value=0, value=10)
        self.show_posts_below(df, threshold)
        # Plots
        st.header(
            "Total Daily Likes",
            help="Shows the total likes received for posts made on each day of the week. This number can increase retroactively as older posts receive new likes.",
            divider="violet",
        )
        chart = Visualizer.plot_likes_per_day(df)
        st.altair_chart(chart, use_container_width=True)
        st.header(
            "Total Hourly Likes",
            help="Displays the total number of likes received for posts made during each hour of the day. This value can increase retroactively as older posts from that hour receive new likes.",
            divider="violet",
        )
        Visualizer.plot_likes_per_hour(df)
        st.header(
            "Likes per Week",
            help="Shows the total likes received for posts made in each respective week. This number can increase retroactively if older posts receive new likes.",
            divider="violet",
        )
        Visualizer.plot_likes_per_week(df)
        st.header(
            "Likes per Month",
            help="Shows the total likes received for posts made in each respective month. This number can increase retroactively if older posts receive new likes.",
            divider="violet",
        )
        Visualizer.plot_likes_per_month(df)
        st.header("Used model distribution", divider="violet")
        model_distribution_plot = self.visualizer.plot_distribution(df, "model")
        st.header("Used sampling method distribution", divider="violet")
        sampling_method_distribution_plot = self.visualizer.plot_distribution(
            df, "sampling_method"
        )
        monthly_likes_df = self.data_transformer.get_monthly_likes(df)
        st.header("🌚Post frequency by Moon Phase🌝", divider="violet")
        self.plot_moon_phase_frequency(df)
        with st.expander("See likes by Week/Month"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Weekly Likes:")
                st.write(self.data_transformer.get_weekly_likes(df))
            with col2:
                st.write("Monthly Likes:")
                st.write(monthly_likes_df)


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
        condition = (~df["nsfw"]) | (df["nsfw"] & nsfw)
        top_posts = (
            df.loc[condition, ["title", "likes", "id"]]
            .sort_values("likes", ascending=False)
            .head(n)
        )

        return top_posts

    @staticmethod
    def get_nsfw_metrics(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"nsfw_count": 0, "nsfw_pct": 0, "nsfw_likes": 0}

        nsfw_mask = df["nsfw"]
        nsfw_count = nsfw_mask.sum()
        total_count = len(df)
        nsfw_likes = df.loc[nsfw_mask, "likes"].sum()

        return {
            "nsfw_count": nsfw_count,
            "nsfw_pct": round(nsfw_count / total_count * 100, 2)
            if total_count > 0
            else np.nan,
            "nsfw_likes": nsfw_likes,
        }

    def standardize_posts(self, data: List[Dict]) -> List[Dict]:
        timestamps = pd.to_datetime(
            [post.get("created_at") for post in data], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        standardized_data = []
        for post, timestamp in zip(data, timestamps):
            photo_media = post.get("photo_media", [])
            moon_phase = get_moon_phase(timestamp)

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
                sampling_steps = [
                    media.get("text_to_image", {}).get("sampling_steps")
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
                "sampling_steps": sampling_steps,
                "moon_phase": moon_phase,
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
    def get_weekly_likes(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        weekly_likes = df.resample("W-Mon").sum()["likes"].reset_index()
        weekly_likes["timestamp"] = weekly_likes["timestamp"] - pd.DateOffset(days=7)
        weekly_likes["week_start"] = weekly_likes["timestamp"]
        weekly_likes["week_end"] = weekly_likes["timestamp"] + pd.DateOffset(days=6)
        weekly_likes["week_range"] = (
            weekly_likes["week_start"].dt.strftime("%b %d")
            + " - "
            + weekly_likes["week_end"].dt.strftime("%b %d %Y")
        )
        weekly_likes.index = weekly_likes["week_range"]
        weekly_likes = weekly_likes[["likes"]]

        return weekly_likes

    def get_monthly_likes(self, df: pd.DataFrame) -> pd.DataFrame:
        resampled_data = df.set_index("timestamp").resample("ME").sum()
        monthly_likes = resampled_data["likes"].reset_index()
        monthly_likes.columns = ["month", "likes"]
        monthly_likes["month"] = pd.to_datetime(
            monthly_likes["month"], utc=True
        ).dt.tz_convert(None)
        formatted_index = monthly_likes["month"].dt.strftime("%b %Y")
        monthly_likes.index = formatted_index
        monthly_likes.drop("month", axis=1, inplace=True)
        return monthly_likes

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
        time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()
        mean_elapsed_minutes = abs(round(time_diffs.mean() / 60))

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

    def plot_likes_per_day(df: pd.DataFrame) -> alt.Chart:
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
        chart = (
            alt.Chart(daily_likes.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("timestamp", title="Day of Week", sort=day_order),
                y=alt.Y("likes", title="Total Likes"),
            )
            .properties(title="Likes per Day of Week")
        )
        return chart

    def plot_likes_per_hour(df: pd.DataFrame):
        timezones = {
            "UTC": "UTC",
            "US/East": "US/Eastern",
            "US/Pacific": "US/Pacific",
            "Brazil/East": "Brazil/East",
            "Europe/Berlin,Warsaw,Budapest": "Europe/Berlin",
            "Europe/Kiev": "Europe/Kiev",
            "Asia/Tokyo": "Asia/Tokyo",
        }
        tz = st.selectbox("Select Timezone", list(timezones.keys()))
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezones[tz])
        hourly_likes = df.groupby(df["timestamp"].dt.hour)["likes"].sum().reset_index()
        hourly_likes.columns = ["hour", "likes"]
        fig = st.bar_chart(hourly_likes.set_index("hour")["likes"])

    @staticmethod
    def plot_likes_per_week(df: pd.DataFrame):
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
        st.line_chart(filtered_df.set_index("week")["likes"])

    def plot_likes_per_month(df: pd.DataFrame):
        resampled_data = df.set_index("timestamp").resample("ME").sum()
        monthly_likes = resampled_data["likes"].reset_index()
        monthly_likes.columns = ["month", "likes"]
        min_timestamp = (
            monthly_likes["month"].min().to_pydatetime().replace(tzinfo=None)
        )
        max_timestamp = monthly_likes["month"].max().to_pydatetime().replace(
            tzinfo=None
        ) + pd.offsets.MonthEnd(0)
        xmin = st.date_input(
            "Start Date",
            value=min_timestamp,
            min_value=min_timestamp,
            max_value=max_timestamp,
        )
        xmax = st.date_input(
            "End Date",
            value=max_timestamp,
            min_value=min_timestamp,
            max_value=max_timestamp,
        )
        monthly_likes["month"] = pd.to_datetime(
            monthly_likes["month"], utc=True
        ).dt.tz_convert(None)
        mask = (monthly_likes["month"] >= pd.to_datetime(xmin)) & (
            monthly_likes["month"] <= pd.to_datetime(xmax)
        )
        filtered_df = monthly_likes.loc[mask]
        current_month = datetime.now().month
        filtered_df = filtered_df[filtered_df["month"].dt.month != current_month]
        st.line_chart(filtered_df.set_index("month")["likes"])

    def plot_distribution(
        self, df: pd.DataFrame, plot_type: str, min_appearances: int = 10
    ):
        if df.empty:
            return None
        if plot_type == "model":
            column_name = "model_display_names"
            x_label = "Model Name"
            title = "Model Distribution"
        elif plot_type == "sampling_method":
            column_name = "sampling_method_display_names"
            x_label = "Sampling Method"
            title = "Sampling Method Distribution"
        else:
            return None
        flattened_names = df[column_name].explode().dropna()
        cleaned_names = self.clean_names(flattened_names)
        name_counts = cleaned_names.value_counts()
        if name_counts.empty:
            return None
        name_counts = name_counts[name_counts >= min_appearances].reset_index()
        name_counts.columns = ["name", "count"]
        chart = (
            alt.Chart(name_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("name:N", sort="-x", title=x_label),
                tooltip=["name", "count"],
            )
            .properties(title=f"{title} (>={min_appearances} uses)")
        )
        st.altair_chart(chart, use_container_width=True)


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
