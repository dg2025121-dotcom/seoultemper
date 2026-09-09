import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="centered")

CSV_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
CUTOFF_YEAR = 2025      # 이 연도까지의 자료만 사용
MIN_OBS_DAYS = 300      # 이 관측일수 미만인 연도는 제외


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
        .reset_index()
    )

    # 기준 기간 이후 자료, 관측일수 부족한 연도 제외
    yearly = yearly[
        (yearly["연도"] <= CUTOFF_YEAR) & (yearly["관측일수"] >= MIN_OBS_DAYS)
    ]
    yearly = yearly.sort_values("연도").reset_index(drop=True)
    return yearly


st.title("🌡️ 서울 기온 예측기")
st.caption(f"기준 기간: ~{CUTOFF_YEAR}년 · 관측일수 {MIN_OBS_DAYS}일 이상인 연도만 사용")

try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들기에 남은 데이터가 충분하지 않습니다.")
    st.stop()

years = yearly["연도"].values.astype(float)
temps = yearly["평균기온"].values.astype(float)

# 회귀 직선 및 상관계수 계산
slope, intercept, r_value, p_value, std_err = stats.linregress(years, temps)

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

st.markdown(
    f"**회귀 직선에 사용된 연도 수:** {n_years}개 &nbsp;|&nbsp; "
    f"**시작 연도:** {start_year}년 &nbsp;|&nbsp; **끝 연도:** {end_year}년"
)
st.markdown(f"**상관계수 (r):** {r_value:.4f}")

st.divider()

selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)
predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style='text-align:center; padding: 20px 0;'>
        <span style='font-size:20px; color:gray;'>{selected_year}년 예상 평균기온</span><br>
        <span style='font-size:64px; font-weight:bold; color:#e63946;'>{predicted_temp:.2f}°C</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 그래프
line_x = np.array([1900, 2100])
line_y = slope * line_x + intercept

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=years, y=temps,
    mode="markers",
    name="연평균기온 (실제)",
    marker=dict(size=8, color="#1d3557"),
))

fig.add_trace(go.Scatter(
    x=line_x, y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="#e63946", width=2, dash="dash"),
))

fig.add_trace(go.Scatter(
    x=[selected_year], y=[predicted_temp],
    mode="markers",
    name=f"{selected_year}년 예측",
    marker=dict(size=14, color="#f4a261", symbol="star", line=dict(width=1, color="black")),
))

fig.update_layout(
    title="서울 연평균기온 산점도 및 회귀 직선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("사용된 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"관측일수": "관측일수(일)"}),
        use_container_width=True,
    )
