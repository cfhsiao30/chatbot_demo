import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from collections import Counter
import os
import json
import requests
from google import genai

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

# ============================================================
# Page config
# ============================================================
st.set_page_config(layout="wide", page_title="尼泊爾旅遊設計師")

# ============================================================
# Color palette (based on #E8EDD0 natural green)
# ============================================================
BG         = "#E8EDD0"
SIDEBAR_BG = "#D4E2B0"
CARD       = "#F5F2E4"
CIRCLE     = "#3D6B4F"
LINE       = "#8B6B3D"
TIME_CLR   = "#C8860A"
TITLE      = "#3D2B1F"
ICON_BG    = "#7A9E7E"
ACCENT     = "#5A8C6E"
WHITE      = "#FFFFFF"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: #FFFFFF; }}
[data-testid="stSidebar"] {{ background-color: #FFFFFF; }}
[data-testid="stSidebar"] .stMarkdown h3 {{ color: {TITLE}; }}

/* ── Day card ── */
.day-card {{
    background: {CARD};
    border-radius: 18px;
    padding: 22px 26px 14px;
    box-shadow: 0 3px 18px rgba(61,43,31,.10);
    margin-bottom: 14px;
}}
.day-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: {TITLE};
    margin-bottom: 18px;
}}

/* ── Horizontal timeline ── */
.timeline {{
    display: flex;
    align-items: flex-start;
    overflow-x: auto;
    padding-bottom: 4px;
}}
.stop-node {{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 100px;
    text-align: center;
    flex-shrink: 0;
}}
.circle {{
    width: 38px; height: 38px;
    border-radius: 50%;
    background: {CIRCLE};
    color: #fff;
    font-size: .95rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 5px;
}}
.node-icon  {{ font-size: 1.25rem; margin-bottom: 3px; }}
.node-name  {{ font-size: .76rem; font-weight: 600; color: {TITLE}; margin-bottom: 2px; }}
.node-time  {{ font-size: .70rem; color: {TIME_CLR}; font-weight: 500; }}
.node-dur   {{ font-size: .66rem; color: #777; }}

/* ── Connector ── */
.connector {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: flex-start;
    padding-top: 19px; min-width: 68px; flex-shrink: 0;
}}
.conn-line {{
    width: 100%;
    border-top: 2.5px dashed {LINE};
}}
.conn-label {{
    font-size: .60rem; color: {LINE};
    margin-top: 3px; white-space: nowrap;
}}

/* ── Stop detail ── */
.stop-detail {{
    background: {WHITE};
    border-radius: 12px;
    padding: 13px 15px;
    margin: 7px 0;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    box-shadow: 0 1px 6px rgba(61,43,31,.06);
}}
.stop-img {{
    width: 136px; min-width: 136px; height: 96px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid {ICON_BG};
}}
.stop-img-ph {{
    width: 136px; min-width: 136px; height: 96px;
    border-radius: 8px;
    background: {ICON_BG}44;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; color: {ICON_BG};
}}
.stop-info h4 {{ margin: 0 0 5px; color: {TITLE}; font-size: .93rem; }}
.stop-info p  {{ margin: 3px 0; font-size: .80rem; color: #555; line-height: 1.4; }}

/* ── Page indicator ── */
.page-ind {{
    text-align: center; font-size: .82rem;
    color: {TITLE}; font-weight: 600; padding: 4px 0;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Pre-bound Wikipedia images (10 confirmed attractions)
# ============================================================
PRESET_IMAGES = {
    "Pokhara":                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Pokhara_Valley.jpg/500px-Pokhara_Valley.jpg",
    "Syambhunath":            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Swayambhunath_2018.jpg/500px-Swayambhunath_2018.jpg",
    "Chitwan National Park":  "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Chitwan_swamp.jpg/500px-Chitwan_swamp.jpg",
    "Bhaktapur Durbar Square":"https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/View_of_Bhaktapur_Durbar_Square.jpg/500px-View_of_Bhaktapur_Durbar_Square.jpg",
    "Pasupatinath Temple":    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Pashupatinath_Temple-2020.jpg/500px-Pashupatinath_Temple-2020.jpg",
    "Everest Base Camp Trek": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg/500px-Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg",
    "Annapurna Base Camp":    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Machapuchre-wow.jpg/500px-Machapuchre-wow.jpg",
    "Langtang":               "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Starry_night_in_Langtang_National_Park.jpg/500px-Starry_night_in_Langtang_National_Park.jpg",
    "Lumbini":                "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/BRP_Lumbini_Mayadevi_temple.jpg/500px-BRP_Lumbini_Mayadevi_temple.jpg",
    "Bardiya National Park":  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Bardiya_02.jpg/500px-Bardiya_02.jpg",
}

TYPE_ICONS = {
    "宗教聖地": "🛕",
    "冒險健行": "🏔️",
    "自然風景": "🌿",
    "文化古蹟": "🏛️",
}


@st.cache_data(ttl=3600)
def fetch_wikimedia_image(name_en: str) -> str:
    """Fetch thumbnail from Wikipedia API for attractions outside PRESET_IMAGES."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "titles": name_en,
                "prop": "pageimages", "format": "json",
                "pithumbsize": 400, "piprop": "thumbnail",
            },
            timeout=6,
        )
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            url = page.get("thumbnail", {}).get("source", "")
            if url:
                return url
    except Exception:
        pass
    return ""


def get_image(attraction_en: str) -> str:
    swapped = st.session_state.get("swap_image", {})
    if attraction_en in swapped:
        return swapped[attraction_en]
    if attraction_en in PRESET_IMAGES:
        return PRESET_IMAGES[attraction_en]
    return fetch_wikimedia_image(attraction_en)


# ============================================================
# Data — RAG database
# ============================================================
@st.cache_resource
def load_data():
    df = pd.DataFrame([
        {"景點名稱":"Syambhunath","景點名稱_中文":"斯瓦揚布佛塔","好評度":"94%","智慧摘要":"強烈推薦","lat":27.7148996,"lng":85.2903957,"最佳造訪季節":"3-5月、9-11月","特色活動或體驗":["猴廟參訪","盆地全景觀賞","佛教文化體驗","觀賞加德滿都全景","參拜佛塔","與猴群互動"],"回答方向提示":"加德滿都的制高點，景色壯觀、猴子可愛","地點類型":"宗教聖地","適合族群建議":["情侶","攝影愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["人潮擁擠","猴子搶食物"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Annapurna Base Camp","景點名稱_中文":"安娜普納基地營","好評度":"87%","智慧摘要":"可安排短暫造訪","lat":28.53,"lng":83.878,"最佳造訪季節":"3-5月、9-11月","特色活動或體驗":["高山健行","雪山觀景","高海拔挑戰","欣賞冰河與雪山景觀","體驗高山文化"],"回答方向提示":"經典登山路線，提醒注意體能與天氣","地點類型":"冒險健行","適合族群建議":["背包客","冒險愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["高海拔引起高山症","天氣變化快"],"關聯景點建議":["Pokhara","Ghorepani Poon Hill"]},
        {"景點名稱":"Bardiya National Park","景點名稱_中文":"巴迪亞國家公園","好評度":"91%","智慧摘要":"值得一遊","lat":28.3648644,"lng":81.5596427,"最佳造訪季節":"11-2月","特色活動或體驗":["叢林野生動物觀察","孟加拉虎與犀牛觀察","叢林探險","野生動物觀察","叢林徒步","搭乘獨木舟"],"回答方向提示":"安靜且原始的國家公園","地點類型":"自然風景","適合族群建議":["野生動物愛好者","家庭"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["交通不便","導遊服務質量參差"],"關聯景點建議":["Chitwan National Park"]},
        {"景點名稱":"Bhaktapur Durbar Square","景點名稱_中文":"巴克塔普爾杜巴廣場","好評度":"94%","智慧摘要":"強烈推薦","lat":27.67207,"lng":85.4282951,"最佳造訪季節":"3-5月、9-11月","特色活動或體驗":["中世紀建築參訪","雕刻藝術欣賞","手工藝文化體驗","探索古老建築","品嚐當地美食","參訪傳統工藝坊"],"回答方向提示":"古城風情與文化遺產","地點類型":"文化古蹟","適合族群建議":["歷史愛好者","家庭"],"平均停留時間建議":"2–3 小時","常見抱怨或負評原因":["建築受損","重建工程中"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Chitwan National Park","景點名稱_中文":"奇特旺國家公園","好評度":"91%","智慧摘要":"值得一遊","lat":27.519285,"lng":84.3135318,"最佳造訪季節":"10–3 月","特色活動或體驗":["騎象觀察動物","叢林探險","獨木舟觀察鳥類","野生動物觀察","叢林騎象","搭乘獨木舟"],"回答方向提示":"適合體驗野生動物與自然生態","地點類型":"自然風景","適合族群建議":["家庭","野生動物愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["觀察野生動物機會不穩定","價格偏高"],"關聯景點建議":["Bardiya National Park"]},
        {"景點名稱":"Everest Base Camp Trek","景點名稱_中文":"聖母峰基地營健行","好評度":"85%","智慧摘要":"口碑普通 建議斟酌","lat":28.0018515,"lng":86.8513119,"最佳造訪季節":"3-5月、9-11月","特色活動或體驗":["高山健行","雪山觀景","高海拔挑戰","欣賞冰河與雪山景觀","體驗高山文化"],"回答方向提示":"挑戰極限的經典健行路線","地點類型":"冒險健行","適合族群建議":["冒險愛好者","健行者"],"平均停留時間建議":"12–14 天","常見抱怨或負評原因":["行程艱辛","住宿條件簡陋"],"關聯景點建議":["Lukla","Namche Bazaar"]},
        {"景點名稱":"Langtang","景點名稱_中文":"朗塘國家公園","好評度":"92%","智慧摘要":"強烈推薦","lat":28.2062873,"lng":85.6229296,"最佳造訪季節":"3-5月、9-11月","特色活動或體驗":["國家公園健行","藏族文化村落","冰川觀景","健行","觀山日出","村落住宿體驗","溫泉放鬆"],"回答方向提示":"景色壯麗、適合初中級登山者","地點類型":"冒險健行","適合族群建議":["健行者","自然愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["交通不便","住宿選擇有限"],"關聯景點建議":["Kathmandu","Gosaikunda Lake"]},
        {"景點名稱":"Lumbini","景點名稱_中文":"藍毗尼","好評度":"94%","智慧摘要":"強烈推薦","lat":27.9207402,"lng":82.7347142,"最佳造訪季節":"10–3 月","特色活動或體驗":["佛教聖地參訪","寺廟探索","冥想靜修","參訪佛教聖地","瞻仰佛塔","冥想與靜修"],"回答方向提示":"心靈沉澱、宗教意義深厚的朝聖感","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["設施簡陋","周邊環境需改善"],"關聯景點建議":["Tansen","Palpa"]},
        {"景點名稱":"Pasupatinath Temple","景點名稱_中文":"帕舒帕提那寺","好評度":"91%","智慧摘要":"強烈推薦","lat":27.710512,"lng":85.3488125,"最佳造訪季節":"10–3 月","特色活動或體驗":["印度教聖地參拜","觀賞火葬儀式","宗教文化體驗","參拜印度教聖地","觀賞火葬儀式","體驗宗教文化"],"回答方向提示":"印度教最重要的聖地之一","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1 小時","常見抱怨或負評原因":["人潮擁擠","香火濃烈"],"關聯景點建議":["Boudhanath Stupa"]},
        {"景點名稱":"Pokhara","景點名稱_中文":"博卡拉","好評度":"94%","智慧摘要":"強烈推薦","lat":28.2095831,"lng":83.9855674,"最佳造訪季節":"10–3 月","特色活動或體驗":["泛舟費瓦湖","安納普爾納日出","滑翔傘","洞窟與瀑布探索","湖上划船","瀑布探訪","溫泉放鬆"],"回答方向提示":"湖光山色、氣氛悠閒，適合放鬆與拍照","地點類型":"自然風景","適合族群建議":["情侶","家庭","攝影愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["部分地區交通擁堵","設施需升級"],"關聯景點建議":["Sarangkot","Phewa Lake"]},
    ])

    def combine_info(row):
        acts = ", ".join(row["特色活動或體驗"]) if isinstance(row["特色活動或體驗"], list) else row["特色活動或體驗"]
        groups = ", ".join(row["適合族群建議"]) if isinstance(row["適合族群建議"], list) else row["適合族群建議"]
        complaints = ", ".join(row["常見抱怨或負評原因"]) if isinstance(row["常見抱怨或負評原因"], list) else row["常見抱怨或負評原因"]
        related = ", ".join(row["關聯景點建議"]) if isinstance(row["關聯景點建議"], list) else row["關聯景點建議"]
        return (
            f"智慧摘要: {row['智慧摘要']}。 好評度: {row['好評度']}。 "
            f"最佳造訪季節: {row['最佳造訪季節']}. 特色活動或體驗: {acts}。 "
            f"地點類型: {row['地點類型']}. 適合族群建議: {groups}. "
            f"平均停留時間建議: {row['平均停留時間建議']}. 常見抱怨或負評原因: {complaints}。 "
            f"關聯景點建議: {related}."
        )

    df["combined_info"] = df.apply(combine_info, axis=1)
    return df


df = load_data()


@st.cache_resource
def build_faiss_index(dataframe):
    emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = emb.encode(dataframe["combined_info"].tolist(), convert_to_numpy=True)
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    return idx, emb


faiss_idx, embedder = build_faiss_index(df)


@st.cache_resource
def load_emotion_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512,
    )


emotion_analyzer = load_emotion_analyzer()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY environment variable not set. Please add it to Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ============================================================
# Itinerary generation
# ============================================================
def generate_itinerary(preferences: list, days: int) -> list:
    prefs = "、".join(preferences) if preferences else "綜合體驗"
    spots = "\n".join(
        f"- {r['景點名稱_中文']}（{r['景點名稱']}）：{r['地點類型']}，停留{r['平均停留時間建議']}"
        for _, r in df.iterrows()
    )
    prompt = f"""你是專業尼泊爾旅遊設計師。設計 {days} 天行程，回傳純 JSON array，不加說明文字或 markdown。

旅遊偏好：{prefs}
總天數：{days}

可選景點（不可超出範圍，不重複）：
{spots}

規則：每天 2-3 個景點；依地理距離排列；從 08:00 起安排時間；估算景點間車程。

格式：
[{{"day":1,"stops":[{{"attraction_en":"Pokhara","attraction_zh":"博卡拉","type":"自然風景","start_time":"08:30","end_time":"11:30","duration":"約3小時","food_tip":"嘗試當地Momo餃子","reminder":"山區早晚溫差大記得帶外套","travel_to_next":"車程約45分"}}]}}]"""

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = resp.text.strip()
    # Strip possible markdown fences
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw.strip())


# ============================================================
# RAG + chat
# ============================================================
def retrieve(query: str, top_k: int = 5):
    vec = embedder.encode([query], convert_to_numpy=True)
    _, I = faiss_idx.search(vec, top_k)
    return df.iloc[I[0]]


def generate_answer(query: str, docs, history: list) -> str:
    context = "\n".join(
        f"{r['景點名稱_中文']}: 好評度{r['好評度']} | {r['地點類型']} | "
        f"停留{r['平均停留時間建議']} | {r['智慧摘要']} | 季節:{r['最佳造訪季節']}"
        for _, r in docs.iterrows()
    )
    memory = "\n".join(
        f"使用者：{h.get('query','')}\n助理：{h.get('answer','')}"
        for h in history[-5:]
    )
    emotion_fb = ""
    if history:
        last = history[-1]
        if isinstance(last, dict) and "emotion_label" in last:
            emotion_fb = f"（上次回答情緒：{last['emotion_label']}，請調整語氣）"

    prompt = f"""你是專業親切的尼泊爾旅遊助理{emotion_fb}。
用中文、Markdown 格式、5-7句話回答。好評度>91%的景點可特別強調。

對話記憶：
{memory}

景點資料：
{context}

問題：{query}"""

    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text.strip()


# ============================================================
# Session state init
# ============================================================
_defaults = {
    "conversation_history": [],
    "emotion_log": [],
    "trend_counter": Counter(),
    "itinerary": None,
    "current_day": 0,
    "swap_image": {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🌿 旅遊偏好")
    preferences = st.multiselect(
        "選擇旅遊類型",
        ["冒險健行", "文化古蹟", "自然風景", "宗教聖地"],
        placeholder="可複選",
    )

    st.markdown("### 📅 規劃進度")
    days = st.number_input("旅程天數", min_value=1, max_value=14, value=3, step=1)

    if st.session_state.itinerary:
        total_d = len(st.session_state.itinerary)
        cur_d = st.session_state.current_day
        st.progress((cur_d + 1) / total_d)
        st.caption(f"瀏覽中 {cur_d + 1}/{total_d} 天")

    st.markdown("### 🗺️ 參考行程")
    if st.button("✨ 生成建議行程", use_container_width=True, type="primary"):
        with st.spinner("AI 正在規劃你的行程..."):
            try:
                result = generate_itinerary(preferences, int(days))
                st.session_state.itinerary = result
                st.session_state.current_day = 0
                st.session_state.swap_image = {}
                st.rerun()
            except Exception as e:
                st.error(f"生成失敗：{e}")

# ============================================================
# Main — Title
# ============================================================
st.markdown(f"""
<div style="margin-bottom:18px">
  <span style="font-size:2rem;font-weight:800;color:{TITLE}">🏔️ 尼泊爾旅遊設計師</span><br>
  <span style="color:{ACCENT};font-size:1rem">打造專屬你的夢幻尼泊爾之旅</span><br>
  <span style="color:{LINE};font-size:.85rem">🎒 旅途準備好了嗎？AI旅伴已打包好地圖與好心情。</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Day card renderer
# ============================================================
def render_day_card(day_data: dict, day_idx: int, total: int):
    stops = day_data.get("stops", [])
    day_num = day_data.get("day", day_idx + 1)

    # Build timeline HTML (static, no interactive widgets)
    nodes_html = ""
    for i, s in enumerate(stops):
        icon = TYPE_ICONS.get(s.get("type", ""), "📍")
        nodes_html += f"""
        <div class="stop-node">
          <div class="circle">{i + 1}</div>
          <div class="node-icon">{icon}</div>
          <div class="node-name">{s.get('attraction_zh', '')}</div>
          <div class="node-time">{s.get('start_time', '')} – {s.get('end_time', '')}</div>
          <div class="node-dur">{s.get('duration', '')}</div>
        </div>"""
        if i < len(stops) - 1:
            nodes_html += f"""
        <div class="connector">
          <div class="conn-line"></div>
          <div class="conn-label">{s.get('travel_to_next', '')}</div>
        </div>"""

    st.markdown(f"""
    <div class="day-card">
      <div class="day-title">🌿 第 {day_num} 天行程</div>
      <div class="timeline">{nodes_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Stop detail rows (image + info)
    for stop in stops:
        en = stop.get("attraction_en", "")
        zh = stop.get("attraction_zh", "")
        icon = TYPE_ICONS.get(stop.get("type", ""), "📍")
        img_url = get_image(en)

        img_tag = (
            f'<img src="{img_url}" class="stop-img" alt="{zh}">'
            if img_url
            else f'<div class="stop-img-ph">{icon}</div>'
        )
        food_html = f"<p>🍜 <b>餐食建議：</b>{stop['food_tip']}</p>" if stop.get("food_tip") else ""
        reminder_html = f"<p>💬 <b>小提醒：</b>{stop['reminder']}</p>" if stop.get("reminder") else ""

        st.markdown(f"""
        <div class="stop-detail">
          {img_tag}
          <div class="stop-info">
            <h4>{icon} {zh}</h4>
            <p>⏰ <b>停留時間：</b>{stop.get('duration', '')}</p>
            {food_html}
            {reminder_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Swap button only for non-preset attractions
        if en not in PRESET_IMAGES:
            if st.button(f"🔄 換圖（{zh}）", key=f"swap_{en}_{day_idx}"):
                new_url = fetch_wikimedia_image(f"{en} Nepal")
                if new_url:
                    st.session_state.swap_image[en] = new_url
                    st.rerun()
                else:
                    st.warning("找不到其他圖片")

    # Left / right navigation
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_l:
        if day_idx > 0:
            if st.button("← 上一天", key=f"prev_{day_idx}"):
                st.session_state.current_day = day_idx - 1
                st.rerun()
    with col_c:
        st.markdown(
            f'<div class="page-ind">第 {day_idx + 1} / {total} 天</div>',
            unsafe_allow_html=True,
        )
    with col_r:
        if day_idx < total - 1:
            if st.button("下一天 →", key=f"next_{day_idx}"):
                st.session_state.current_day = day_idx + 1
                st.rerun()


# ============================================================
# Main — Itinerary or placeholder
# ============================================================
if st.session_state.itinerary:
    itin = st.session_state.itinerary
    cur = st.session_state.current_day
    cur = max(0, min(cur, len(itin) - 1))

    render_day_card(itin[cur], cur, len(itin))

    # Map
    st.markdown(f"<h4 style='color:{TITLE};margin-top:14px'>🗺️ 行程地圖</h4>", unsafe_allow_html=True)

    if FOLIUM_OK:
        pts = []
        for s in itin[cur].get("stops", []):
            row = df[df["景點名稱"] == s.get("attraction_en", "")]
            if not row.empty:
                r = row.iloc[0]
                pts.append({"name": s.get("attraction_zh", ""), "lat": r["lat"], "lng": r["lng"]})

        if pts:
            clat = sum(p["lat"] for p in pts) / len(pts)
            clng = sum(p["lng"] for p in pts) / len(pts)
            m = folium.Map(location=[clat, clng], zoom_start=8, tiles="OpenStreetMap")
            for i, p in enumerate(pts):
                folium.Marker(
                    [p["lat"], p["lng"]],
                    tooltip=f"{i + 1}. {p['name']}",
                    popup=p["name"],
                    icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
                ).add_to(m)
            st_folium(m, height=380, use_container_width=True)
        else:
            st.info("本日景點無法對應地圖座標")
    else:
        st.info("請安裝 `folium` 與 `streamlit-folium` 以顯示互動地圖。")

else:
    st.markdown(f"""
    <div style="background:{CARD};border-radius:16px;padding:40px;
                text-align:center;color:{LINE};margin:20px 0;
                box-shadow:0 2px 12px rgba(61,43,31,.08)">
      🌿 請在左側選擇旅遊偏好與天數，點擊「生成建議行程」開始規劃。
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Chat interface
# ============================================================
st.markdown(
    f"<hr style='border-color:{LINE}44;margin:22px 0 10px'>"
    f"<h4 style='color:{TITLE}'>💬 旅遊問題？問我吧！</h4>",
    unsafe_allow_html=True,
)

for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("輸入你的旅遊問題..."):
    st.session_state.conversation_history.append(
        {"role": "user", "content": query, "query": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                docs = retrieve(query)
                answer = generate_answer(query, docs, st.session_state.conversation_history)
                result = emotion_analyzer(answer[:2000])[0]
                label, score = result["label"], result["score"]
                st.session_state.emotion_log.append(label)
                st.session_state.trend_counter = Counter(st.session_state.emotion_log)
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "answer": answer,
                    "emotion_label": label,
                    "emotion_score": score,
                    "source": docs["景點名稱_中文"].tolist(),
                    "trend_snapshot": dict(st.session_state.trend_counter),
                })
                st.markdown(answer)
            except Exception as e:
                err = f"發生錯誤：{e}"
                st.error(err)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": err}
                )
