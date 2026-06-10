import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from collections import Counter
import os
import json
import re
import requests
from pathlib import Path
from google import genai
from google.genai import types

import base64

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(layout="wide", page_title="尼泊爾旅遊設計師")

st.markdown("""
<style>
/* ── 隱藏預設 header 空白 ── */
[data-testid="stHeader"] { display: none; }
.main > div { padding-top: 0 !important; }
[data-testid="stMainBlockContainer"] {
    padding-top: 8px !important;
    padding-bottom: 0 !important;
    max-height: 100vh;
    overflow: hidden;
}

/* ── 右欄背景 ── */
[data-testid="column"]:last-child {
    background: #fafaf9;
}

/* ── 兩欄等高，撐滿視窗 ── */
[data-testid="column"] {
    height: calc(100vh - 20px) !important;
    overflow: hidden !important;
}

/* ── 進度條綠色 ── */
div[data-testid="stProgressBar"] > div > div > div {
    background-color: #5B8A5B !important;
}

/* ── 輸入框樣式 ── */
.stForm [data-testid="stTextArea"] textarea {
    border-radius: 10px !important;
    border: 1.5px solid #d1d5db !important;
    resize: none !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}
.stForm [data-testid="stTextArea"] textarea:focus {
    border-color: #5B8A5B !important;
    box-shadow: 0 0 0 2px rgba(91,138,91,0.15) !important;
}

/* ── 送出按鈕 ── */
.stForm [data-testid="stFormSubmitButton"] button {
    background: #5B8A5B !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}
.stForm [data-testid="stFormSubmitButton"] button:hover {
    background: #4a7a4a !important;
}

/* ── 對話歷史捲動容器內的 chat_message 間距 ── */
[data-testid="stVerticalBlock"] [data-testid="stChatMessage"] {
    padding: 4px 0 !important;
}

/* ── 全域 chat_input 隱藏（改用 form）── */
[data-testid="stChatInput"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1️⃣ 資料載入
# ============================================================
@st.cache_resource
def load_data():
    df = pd.DataFrame([
        {"景點名稱":"Syambhunath","景點名稱_中文":"斯瓦揚布佛塔","好評度":"94%","評價比例":"878/35/17/0","智慧摘要":"強烈推薦","lat":27.7148996,"lng":85.2903957,"types":["tourist_attraction","place_of_worship","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；天氣涼爽","特色活動或體驗":["猴廟參訪","盆地全景觀賞","佛教文化體驗","觀賞加德滿都全景","參拜佛塔","與猴群互動"],"回答方向提示":"可形容為「加德滿都的制高點，景色壯觀、猴子可愛」","地點類型":"宗教聖地","適合族群建議":["情侶","攝影愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["人潮擁擠","猴子搶食物"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Annapurna Base Camp","景點名稱_中文":"安娜普納基地營","好評度":"87%","評價比例":"115/5/12/0","智慧摘要":"可安排短暫造訪","lat":28.53,"lng":83.878,"types":["natural_feature","establishment"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["生態區域穿越","高山景觀觀賞","健行","高山健行","露營","欣賞雪山日出"],"回答方向提示":"可描述「經典登山路線」，提醒注意體能與天氣","地點類型":"冒險健行","適合族群建議":["背包客","冒險愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["高海拔引起高山症","天氣變化快"],"關聯景點建議":["Pokhara","Ghorepani Poon Hill"]},
        {"景點名稱":"Bardiya National Park","景點名稱_中文":"巴迪亞國家公園","好評度":"91%","評價比例":"541/29/25/0","智慧摘要":"值得一遊","lat":28.3648644,"lng":81.5596427,"types":["tourist_attraction","park","point_of_interest","establishment"],"最佳造訪季節":"11-2月；乾季","特色活動或體驗":["叢林野生動物觀察","孟加拉虎與犀牛觀察","叢林探險","野生動物觀察","叢林徒步","搭乘獨木舟"],"回答方向提示":"可提「安靜且原始的國家公園」，適合熱愛野生自然的旅人","地點類型":"自然風景","適合族群建議":["野生動物愛好者","家庭"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["交通不便","導遊服務質量參差"],"關聯景點建議":["Chitwan National Park"]},
        {"景點名稱":"Bhaktapur Durbar Square","景點名稱_中文":"巴克塔普爾杜巴廣場","好評度":"94%","評價比例":"874/29/27/0","智慧摘要":"強烈推薦","lat":27.67207,"lng":85.4282951,"types":["tourist_attraction","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["中世紀建築參訪","雕刻藝術欣賞","手工藝文化體驗","探索古老建築","品嚐當地美食","參訪傳統工藝坊"],"回答方向提示":"可強調「古城風情與文化遺產」，適合愛歷史或攝影的人","地點類型":"文化古蹟","適合族群建議":["歷史愛好者","家庭"],"平均停留時間建議":"2–3 小時","常見抱怨或負評原因":["建築受損","重建工程中"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Chitwan National Park","景點名稱_中文":"奇特旺國家公園","好評度":"91%","評價比例":"844/48/38/0","智慧摘要":"值得一遊","lat":27.519285,"lng":84.3135318,"types":["tourist_attraction","park","point_of_interest","establishment"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["騎象觀察動物","叢林探險","獨木舟觀察鳥類","野生動物觀察","叢林騎象","搭乘獨木舟"],"回答方向提示":"可描述為「適合體驗野生動物與自然生態的地方」","地點類型":"自然風景","適合族群建議":["家庭","野生動物愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["觀察野生動物機會不穩定","價格偏高"],"關聯景點建議":["Bardiya National Park"]},
        {"景點名稱":"Everest Base Camp Trek","景點名稱_中文":"聖母峰基地營健行","好評度":"85%","評價比例":"1994/3/14/0","智慧摘要":"口碑普通 建議斟酌或查更多資訊","lat":28.0018515,"lng":86.8513119,"types":["tourist_attraction","travel_agency","park","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；冬季極寒","特色活動或體驗":["高山健行","雪山觀景","高海拔挑戰","欣賞冰河與雪山景觀","體驗高山文化"],"回答方向提示":"可提「挑戰極限的經典健行路線」，但非所有人都適合","地點類型":"冒險健行","適合族群建議":["冒險愛好者","健行者"],"平均停留時間建議":"12–14 天","常見抱怨或負評原因":["行程艱辛","住宿條件簡陋"],"關聯景點建議":["Lukla","Namche Bazaar"]},
        {"景點名稱":"Langtang","景點名稱_中文":"朗塘國家公園","好評度":"92%","評價比例":"622/27/24/0","智慧摘要":"強烈推薦","lat":28.2062873,"lng":85.6229296,"types":["locality","political"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["國家公園健行","藏族文化村落","冰川觀景","健行","觀山日出","村落住宿體驗","溫泉放鬆"],"回答方向提示":"可強調「景色壯麗、適合初中級登山者」","地點類型":"冒險健行","適合族群建議":["健行者","自然愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["交通不便","住宿選擇有限"],"關聯景點建議":["Kathmandu","Gosaikunda Lake"]},
        {"景點名稱":"Lumbini","景點名稱_中文":"藍毗尼","好評度":"94%","評價比例":"878/28/29/0","智慧摘要":"強烈推薦","lat":27.9207402,"lng":82.7347142,"types":["administrative_area_level_1","political"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["佛教聖地參訪","寺廟探索","冥想靜修","參訪佛教聖地","瞻仰佛塔","冥想與靜修"],"回答方向提示":"可強調「心靈沉澱、宗教意義深厚」的朝聖感","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["設施簡陋","周邊環境需改善"],"關聯景點建議":["Tansen","Palpa"]},
        {"景點名稱":"Pasupatinath Temple","景點名稱_中文":"帕舒帕提那寺","好評度":"91%","評價比例":"849/33/48/0","智慧摘要":"強烈推薦","lat":27.710512,"lng":85.3488125,"types":["hindu_temple","place_of_worship","point_of_interest","establishment"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["印度教聖地參拜","觀賞火葬儀式","宗教文化體驗","參拜印度教聖地","觀賞火葬儀式","體驗宗教文化"],"回答方向提示":"可提「印度教最重要的聖地之一」，但提醒人潮較多","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1 小時","常見抱怨或負評原因":["人潮擁擠","香火濃烈"],"關聯景點建議":["Boudhanath Stupa","Pashupatinath Temple"]},
        {"景點名稱":"Pokhara","景點名稱_中文":"博卡拉","好評度":"94%","評價比例":"880/45/11/0","智慧摘要":"強烈推薦","lat":28.2095831,"lng":83.9855674,"types":["locality","political"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["泛舟費瓦湖","安納普爾納日出","滑翔傘","洞窟與瀑布探索","湖上划船","瀑布探訪","滑翔傘","溫泉放鬆"],"回答方向提示":"可描述成「湖光山色、氣氛悠閒的城市」，適合放鬆與拍照","地點類型":"自然風景","適合族群建議":["情侶","家庭","攝影愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["部分地區交通擁堵","設施需升級"],"關聯景點建議":["Sarangkot","Phewa Lake"]},
    ])

    useful_columns = ['好評度','最佳造訪季節','特色活動或體驗','地點類型','適合族群建議','平均停留時間建議','常見抱怨或負評原因','關聯景點建議']

    def combine_info(row):
        info = f"智慧摘要: {row['智慧摘要']}。"
        info += f" 好評度: {row['好評度']}。"
        info += f" 最佳造訪季節: {row['最佳造訪季節']}."
        if isinstance(row['特色活動或體驗'], list):
            info += f" 特色活動或體驗: {', '.join(row['特色活動或體驗'])}。"
        info += f" 地點類型: {row['地點類型']}."
        if isinstance(row['適合族群建議'], list):
            info += f" 適合族群建議: {', '.join(row['適合族群建議'])}."
        info += f" 平均停留時間建議: {row['平均停留時間建議']}."
        if isinstance(row['常見抱怨或負評原因'], list):
            info += f" 常見抱怨或負評原因: {', '.join(row['常見抱怨或負評原因'])}。"
        if isinstance(row['關聯景點建議'], list):
            info += f" 關聯景點建議: {', '.join(row['關聯景點建議'])}."
        return info

    df['combined_info'] = df.apply(combine_info, axis=1)
    return df

df = load_data()

# ============================================================
# 2️⃣ FAISS 向量索引
# ============================================================
@st.cache_resource
def build_faiss_index(dataframe):
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.encode(dataframe["combined_info"].tolist(), convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embedder

index, embedder = build_faiss_index(df)

# ============================================================
# 3️⃣ 情緒分析模型
# ============================================================
@st.cache_resource
def load_emotion_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512,
    )

emotion_analyzer = load_emotion_analyzer()

def analyze_emotion(text):
    result = emotion_analyzer(text[:2000])[0]
    return result['label'], result['score']

# ============================================================
# Gemini 客戶端
# ============================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY environment variable not set. Please add it to your Streamlit Cloud secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ============================================================
# 4️⃣ 景點圖片
# 優先順序：① 本地 images/（開發者覆蓋） ② Wikipedia API ③ Wikimedia 備援
# ============================================================
# Wikipedia 英文文章標題對照（用於自動抓取縮圖）
WIKIPEDIA_TITLES = {
    "斯瓦揚布佛塔":      "Swayambhunath",
    "安娜普納基地營":     "Annapurna_Base_Camp",
    "巴迪亞國家公園":     "Bardiya_National_Park",
    "巴克塔普爾杜巴廣場": "Bhaktapur_Durbar_Square",
    "奇特旺國家公園":     "Chitwan_National_Park",
    "聖母峰基地營健行":   "Everest_Base_Camp_trek",
    "朗塘國家公園":       "Langtang_National_Park",
    "藍毗尼":             "Lumbini",
    "帕舒帕提那寺":       "Pashupatinath_Temple",
    "博卡拉":             "Pokhara",
}

WIKIMEDIA_IMAGES = {
    "博卡拉":           "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Pokhara_Valley.jpg/330px-Pokhara_Valley.jpg",
    "斯瓦揚布佛塔":     "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Swayambhunath_2018.jpg/330px-Swayambhunath_2018.jpg",
    "巴克塔普爾杜巴廣場":"https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/View_of_Bhaktapur_Durbar_Square.jpg/330px-View_of_Bhaktapur_Durbar_Square.jpg",
    "帕舒帕提那寺":     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Pashupatinath_Temple-2020.jpg/330px-Pashupatinath_Temple-2020.jpg",
    "奇特旺國家公園":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Chitwan_swamp.jpg/330px-Chitwan_swamp.jpg",
    "安娜普納基地營":   "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Under_stars_and_snows.jpg/330px-Under_stars_and_snows.jpg",
    "聖母峰基地營健行": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Khumbutse.jpg/330px-Khumbutse.jpg",
    "朗塘國家公園":     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Starry_night_in_Langtang_National_Park.jpg/330px-Starry_night_in_Langtang_National_Park.jpg",
    "藍毗尼":           "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/BRP_Lumbini_Mayadevi_temple.jpg/330px-BRP_Lumbini_Mayadevi_temple.jpg",
    "巴迪亞國家公園":   "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Bardiya_02.jpg/330px-Bardiya_02.jpg",
}

_COMMONS_SKIP = {"map", "plan", "diagram", "logo", "flag", "icon", "chart",
                 "graph", "template", "seal", "coat", "symbol", "emblem", "sign"}

def _is_photo_url(url: str) -> bool:
    """過濾非照片檔案：排除 SVG 及地圖/示意圖等。"""
    lower = url.lower()
    if lower.endswith(".svg"):
        return False
    filename = lower.rsplit("/", 1)[-1]
    return not any(kw in filename for kw in _COMMONS_SKIP)


@st.cache_data(show_spinner=False)
def fetch_commons_images(search_query: str, fallback_parent: str = "", limit: int = 6) -> list[str]:
    """搜尋 Wikimedia Commons，回傳最多 limit 個候選圖片 URL（list）。"""
    results = []
    for query in dict.fromkeys([search_query, WIKIPEDIA_TITLES.get(fallback_parent, "")]):
        if not query:
            continue
        try:
            params = {
                "action": "query",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": query,
                "gsrlimit": str(limit * 2),   # 多抓再過濾
                "prop": "imageinfo",
                "iiprop": "url|size",
                "iiurlwidth": "600",
                "format": "json",
                "origin": "*",
            }
            r = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                timeout=8,
                headers={"User-Agent": "NepalTravelApp/1.0"},
            )
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
                    info = page.get("imageinfo", [{}])[0]
                    url = info.get("thumburl") or info.get("url", "")
                    if url and _is_photo_url(url) and url not in results:
                        results.append(url)
                        if len(results) >= limit:
                            return results
        except Exception:
            continue
    return results


def load_spot_image(search_query: str = "", parent: str = "", used_urls: set | None = None):
    """
    圖片來源優先順序：
    ① Wikimedia Commons（以 search_query 精準搜，跳過 used_urls 中已用過的）
    ② 本地 images/
    ③ Wikimedia 硬編碼備援（同樣跳過已用過的）
    回傳 (來源, URL)。
    """
    if used_urls is None:
        used_urls = set()

    candidates = fetch_commons_images(search_query, parent)
    for url in candidates:
        if url not in used_urls:
            return 'web', url

    # 本地圖片（不去重，本來就是精準對應）
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        path = Path(f"images/{parent}{ext}")
        if path.exists():
            return 'local', str(path)

    # 硬編碼備援：依序找未用過的
    if parent in WIKIMEDIA_IMAGES:
        url = WIKIMEDIA_IMAGES[parent]
        if url not in used_urls:
            return 'wikimedia', url

    return None, None


def is_best_season(season_str: str, month_num: int) -> bool:
    """回傳 True 代表 month_num 落在最佳造訪季節內。"""
    ranges = re.findall(r'(\d+)[–\-](\d+)\s*月', season_str)
    good_months: set = set()
    for start, end in ranges:
        s, e = int(start), int(end)
        if s <= e:
            good_months.update(range(s, e + 1))
        else:
            good_months.update(range(s, 13))
            good_months.update(range(1, e + 1))
    if not good_months:
        return True
    return month_num in good_months


# ============================================================
# 5️⃣ 行程生成（Gemini）
# ============================================================
TRIP_TYPE_MAP = {
    "冒險健行": ["安娜普納基地營", "聖母峰基地營健行", "朗塘國家公園"],
    "文化宗教": ["斯瓦揚布佛塔", "帕舒帕提那寺", "藍毗尼", "巴克塔普爾杜巴廣場"],
    "自然生態": ["博卡拉", "奇特旺國家公園", "巴迪亞國家公園"],
}

# ============================================================
# Gemini 統一呼叫：retry × 3 + fallback to gemini-1.5-flash
# ============================================================
import time

_PRIMARY_MODEL   = "gemini-2.5-flash"
_FALLBACK_MODEL  = "gemini-1.5-flash"

def gemini_call(prompt: str, json_mode: bool = False, max_retries: int = 3) -> str:
    """
    呼叫 Gemini，自動 retry（指數退避）。
    三次全失敗後 fallback 到 gemini-1.5-flash 再試一次。
    回傳 response.text（str）。
    """
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    kwargs = {"config": cfg} if cfg else {}

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=_PRIMARY_MODEL, contents=prompt, **kwargs
            )
            return resp.text
        except Exception as e:
            last_err = e
            wait = 2 ** attempt          # 1s, 2s, 4s
            time.sleep(wait)

    # Fallback
    try:
        resp = client.models.generate_content(
            model=_FALLBACK_MODEL, contents=prompt, **kwargs
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(
            f"Gemini 主模型（{_PRIMARY_MODEL}）與備援模型（{_FALLBACK_MODEL}）均失敗。\n"
            f"主模型錯誤：{last_err}\n備援錯誤：{e}"
        )


def generate_itinerary(trip_type: str, total_days: int, travel_month: str = "", companion: str = "", history: list = None, must_visit: list = None) -> dict:
    priority_spots = TRIP_TYPE_MAP.get(trip_type, [])
    all_spots_info = "\n".join([
        f"- {row['景點名稱_中文']}（{row['地點類型']}，停留：{row['平均停留時間建議']}，最佳季節：{row['最佳造訪季節']}）"
        for _, row in df.iterrows()
    ])
    priority_str = "、".join(priority_spots)

    valid_names = [row['景點名稱_中文'] for _, row in df.iterrows()]
    valid_names_str = "、".join(valid_names)

    month_hint = f"出發月份：{travel_month}，請優先安排當月屬於最佳造訪季節的景點。" if travel_month else ""

    companion_guide = {
        "親子同遊": "請避免高強度健行景點（如聖母峰基地營），優先安排適合兒童的景點。",
        "長輩同行": "請避免高海拔、高強度景點，優先安排交通方便、步行量少的景點。",
        "情侶同遊": "可優先安排景色優美、氣氛浪漫的景點。",
    }.get(companion, "")

    chat_hint = ""
    if history:
        recent = [h for h in history[-6:] if h.get("content")]
        if recent:
            lines = "\n".join(
                f"{'使用者' if h['role'] == 'user' else '助理'}：{h['content'][:120]}"
                for h in recent
            )
            chat_hint = f"對話中旅人提到的偏好（請參考）：\n{lines}"

    must_hint = ""
    if must_visit:
        must_hint = f"【必訪景點】旅客明確指定以下景點，行程中必須全部安排，不得省略：{'、'.join(must_visit)}\n"

    prompt = f"""你是尼泊爾旅遊專家。請為旅客規劃 {total_days} 天行程。

【旅客資訊】
偏好類型：{trip_type}
旅伴類型：{companion}
{month_hint}
{companion_guide}

{must_hint}
{chat_hint}

【行程規劃原則】
- 請以「具體活動或體驗」為單位安排行程，而非重複填寫同一個景點大名。
  例如：填「吉普車叢林巡遊」而非重複填「奇特旺國家公園」。
- 整趟行程每項活動不得重複出現。
- 若偏好類型景點不足以填滿天數，可跨類型補充，讓行程豐富完整。

【欄位規則】
- name：具體活動或體驗名稱，可自由描述（例：「犀牛河獨木舟觀鳥」、「斯瓦揚布佛塔日出參拜」）
- parent：必須從以下清單選一個最相關的景點名稱（用於資料對應，必須完全一致）：
  {valid_names_str}
- search_query：2–4 個英文關鍵字，用於搜尋此活動的代表照片。規則：
  · 必須使用具體地名或景觀名詞，不用動詞或抽象詞
  · 交通/移動類活動：用目的地景觀名，例如 "Chitwan jungle Nepal"
  · 自然活動：用地點+景物，例如 "Phewa Lake Pokhara"、"Rapti River Chitwan"
  · 文化/宗教活動：用建築或儀式名，例如 "Swayambhunath temple"、"Pashupatinath cremation ghat"
  · 健行活動：用山名或健行路線，例如 "Annapurna trek mountain"、"Langtang valley glacier"

各景點參考資訊：
{all_spots_info}

每天安排 2–3 項活動。回傳格式（純 JSON）：
{{
  "days": [
    {{
      "day": 1,
      "spots": [
        {{
          "name": "具體活動名稱",
          "parent": "對應景點（必須符合上方清單）",
          "search_query": "English keywords for image search",
          "duration": "建議停留時間",
          "food": "餐食建議（一句話）",
          "tip": "小提醒（一句話）"
        }}
      ]
    }}
  ]
}}"""

    raw = gemini_call(prompt, json_mode=True)
    return json.loads(raw)

# ============================================================
# 6️⃣ RAG 檢索與回答生成
# ============================================================
def retrieve(query, top_k=5):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    return df.iloc[I[0]]

def generate_answer(query, retrieved_docs, history):
    context = "\n".join(
        [f"""{row['景點名稱_中文']}:
好評度: {row['好評度']}
最佳造訪季節: {row['最佳造訪季節']}
特色活動或體驗: {', '.join(row['特色活動或體驗']) if isinstance(row['特色活動或體驗'], list) else row['特色活動或體驗']}
地點類型: {row['地點類型']}
適合族群建議: {', '.join(row['適合族群建議']) if isinstance(row['適合族群建議'], list) else row['適合族群建議']}
平均停留時間建議: {row['平均停留時間建議']}
常見抱怨或負評原因: {', '.join(row['常見抱怨或負評原因']) if isinstance(row['常見抱怨或負評原因'], list) else row['常見抱怨或負評原因']}
關聯景點建議: {', '.join(row['關聯景點建議']) if isinstance(row['關聯景點建議'], list) else row['關聯景點建議']}
智慧摘要: {row['智慧摘要']}
""" for _, row in retrieved_docs.iterrows()]
    )

    memory = "\n".join([
        f"使用者：{h.get('query', '')}\n助理：{h.get('answer', '')}"
        for h in history[-5:]
    ])

    emotion_feedback = ""
    if history:
        last_turn = history[-1]
        if isinstance(last_turn, dict):
            emotion_label = last_turn.get('emotion_label', '未知')
            score = last_turn.get('emotion_score', 0.0)
            emotion_feedback = f"請注意，上一次回答的情緒是 {emotion_label} (信心值 {score:.3f})，請在本次回答中保持親切、專業的語氣，並依據對話歷史來調整回應風格。"

    prompt = f"""
你是一位專業且親切的尼泊爾旅遊助理，就像貼心旅伴。請依照以下指示回答使用者問題：

 **整合資訊**
 - 請務必全面整合檢索到的景點資訊，包含「好評度」、「最佳造訪季節」、「特色活動或體驗」、「地點類型」、「適合族群建議」、「平均停留時間建議」、「常見抱怨或負評原因」和「關聯景點建議」。
 - 將資訊整理成完整、個人化的旅遊建議。

 **回答重點**
 - 提供景點推薦、行程規劃、交通建議、天氣提示、住宿選擇、小提醒。
 - 若使用者提到偏好，請優先考慮並明確回應。
 - 簡要說明不同景點間的地理位置或交通考量，幫助規劃順遊路線。

 **對話連貫性**
 - 結合對話歷史，保持回答連貫，避免前後矛盾。

 **格式與語氣**
 - 使用 Markdown 或清單排版，清楚呈現資訊。
 - 中文回答，保持自然、不誇張、親切且專業的語氣。
 - 5-7 句話內提供建議，避免重複或制式回答。
 - 好評度 > 91% 的景點可特別強調，低於 91% 則不用強調。

 **規劃意圖處理**
 - 若使用者的問題包含「規劃行程」、「安排行程」、「幫我排」等規劃請求，請【不要】自行列出行程，
   改為簡短確認你已掌握的偏好摘要（旅伴、偏好類型、天數、月份），
   並告知「請在右側確認後點『生成專屬行程』，我會為你生成完整行程卡片」。

以下是對話歷史：
{memory}

檢索到的景點資訊：
{context}

使用者問題：
{query}
"""
    return gemini_call(prompt).strip()

# ============================================================
# 7️⃣ 行程卡片顯示
# ============================================================
def _img_src(source: str, img: str) -> str:
    """將本地圖片轉為 base64 data URI；Wikimedia URL 直接回傳。"""
    if source == 'local':
        import base64
        ext = Path(img).suffix.lstrip('.').lower()
        mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
        with open(img, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{data}"
    return img


def _spot_card_html(spot: dict, travel_month_num: int = 0, used_urls: set | None = None) -> str:
    """橫式行程小卡：左圖 | 右文 | 下主標"""
    if used_urls is None:
        used_urls = set()

    name = spot['name']
    parent = spot.get('parent', name)
    search_query = spot.get('search_query', name)
    source, img = load_spot_image(search_query, parent, used_urls)

    # 記錄已使用的 URL，避免後續卡片重複
    if img and source in ('web', 'wikimedia'):
        used_urls.add(img)

    season_warning = False
    if travel_month_num:
        match = df[df['景點名稱_中文'] == parent]
        if not match.empty:
            season_warning = not is_best_season(match.iloc[0]['最佳造訪季節'], travel_month_num)

    # 左側圖片
    if img:
        src = _img_src(source, img)
        credit = '<p style="font-size:9px;color:#aaa;margin:2px 0 0 0;text-align:center;">© Wikipedia CC BY-SA</p>' if source in ('web', 'wikimedia') else ''
        img_block = (
            f'<div style="flex:0 0 140px;width:140px;">'
            f'<img src="{src}" style="width:140px;height:120px;object-fit:cover;'
            f'border-radius:8px 0 0 0;display:block;">'
            f'{credit}'
            f'</div>'
        )
    else:
        img_block = (
            f'<div style="flex:0 0 140px;width:140px;height:120px;background:#D4CFC7;'
            f'border-radius:8px 0 0 0;display:flex;align-items:center;justify-content:center;'
            f'color:#6B6B6B;font-size:11px;flex-direction:column;">📷<br>暫無圖片</div>'
        )

    # 右側文字
    warning_html = (
        '<p style="margin:4px 0 0 0;color:#92650a;font-size:10px;background:#fff3cd;'
        'padding:2px 5px;border-radius:3px;display:inline-block;">⚠️ 非最佳造訪季節</p>'
    ) if season_warning else ''

    text_block = (
        f'<div style="flex:1;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;">'
        f'<div>'
        f'<p style="margin:0 0 5px 0;color:#555;font-size:12px;">⏱️ {spot["duration"]}</p>'
        f'<p style="margin:0 0 5px 0;color:#555;font-size:12px;">🍽️ {spot["food"]}</p>'
        f'<p style="margin:0;color:#E07B39;font-size:12px;">💡 {spot["tip"]}</p>'
        f'</div>'
        f'{warning_html}'
        f'</div>'
    )

    # 底部主標（活動名稱）
    footer = (
        f'<div style="padding:6px 10px 8px 10px;background:#E8F0E8;border-top:1px solid #d8d0c4;">'
        f'<p style="margin:0;font-weight:bold;color:#3D5A3D;font-size:13px;">📍 {name}</p>'
        f'</div>'
    )

    return (
        f'<div style="width:100%;background:#F5F0E8;border-radius:8px;'
        f'border:1px solid #d8d0c4;overflow:hidden;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:stretch;min-height:120px;">'
        f'{img_block}'
        f'{text_block}'
        f'</div>'
        f'{footer}'
        f'</div>'
    )


def show_itinerary_cards(itinerary_data: dict, travel_month_num: int = 0):
    used_urls: set = set()   # 整趟行程共用，跨天去重
    for day_data in itinerary_data['days']:
        day_num = day_data['day']
        st.markdown(
            f'<div style="background:#5B8A5B;color:white;padding:7px 14px;'
            f'border-radius:6px;margin:12px 0 6px 0;font-weight:bold;">📅 第 {day_num} 天</div>',
            unsafe_allow_html=True,
        )
        cards_html = "".join(
            _spot_card_html(spot, travel_month_num, used_urls)
            for spot in day_data['spots']
        )
        st.markdown(cards_html, unsafe_allow_html=True)



# ============================================================
# 8️⃣ 匯出：自包含 HTML（喜好摘要 + 行程小卡 + base64 圖片）
# ============================================================
def build_export_html(itinerary_data: dict, params: dict, travel_month_num: int = 0) -> str:
    """生成可離線瀏覽的自包含 HTML 字串。"""

    # ── 喜好摘要區塊 ────────────────────────────────────────
    companion  = params.get('companion')  or params.get('confirm_companion') or '未指定'
    trip_type  = params.get('trip_type')  or params.get('confirm_trip_type') or '未指定'
    month      = params.get('travel_month') or '未指定'
    days_count = len(itinerary_data.get('days', []))

    tag_html = "".join(
        f'<span style="background:#e8f0e8;color:#3d5a3d;padding:3px 12px;'
        f'border-radius:12px;font-size:13px;margin-right:8px;">{t}</span>'
        for t in [f"👥 {companion}", f"🧭 {trip_type}", f"🗓️ {month}出發", f"⏱️ {days_count} 天"]
    )

    # ── 行程卡片 HTML（複用 _spot_card_html）───────────────
    cards_html = ""
    used_urls_export: set = set()
    for day_data in itinerary_data.get('days', []):
        cards_html += (
            f'<div style="background:#5B8A5B;color:white;padding:8px 16px;'
            f'border-radius:6px;margin:16px 0 8px 0;font-weight:bold;font-size:15px;">'
            f'📅 第 {day_data["day"]} 天</div>'
        )
        for spot in day_data.get('spots', []):
            cards_html += _spot_card_html(spot, travel_month_num, used_urls_export)

    # ── 完整 HTML ───────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>尼泊爾行程 · {days_count} 天 · {month}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Noto Sans TC", sans-serif;
          background: #f7f5f0; color: #1a1a1a; padding: 24px 16px; }}
  .container {{ max-width: 780px; margin: 0 auto; }}
  header {{ background: #3D5A3D; color: white; padding: 20px 24px;
            border-radius: 12px; margin-bottom: 24px; }}
  header h1 {{ font-size: 22px; margin-bottom: 6px; }}
  header p  {{ font-size: 13px; opacity: .8; margin-bottom: 12px; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  @media print {{
    body {{ background: white; padding: 0; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏔️ 尼泊爾專屬行程</h1>
    <p>由 AI 旅遊設計師根據您的偏好量身規劃</p>
    <div class="tags">{tag_html}</div>
  </header>
  {cards_html}
  <p style="text-align:center;color:#aaa;font-size:11px;margin-top:24px;">
    由尼泊爾旅遊設計師生成 · 圖片來源 Wikipedia CC BY-SA
  </p>
</div>
</body>
</html>"""


# ============================================================
# 行程觸發：關鍵字偵測 + 參數萃取
# ============================================================
_ITINERARY_KEYWORDS = [
    "規劃行程", "生成行程", "安排行程", "幫我排", "行程規劃",
    "幫我規劃", "幫我安排", "排行程", "出行程", "做行程", "產生行程", "規劃一下",
]

def wants_itinerary(query: str) -> bool:
    return any(kw in query for kw in _ITINERARY_KEYWORDS)


def extract_trip_params(history: list) -> dict:
    recent = [h for h in history[-12:] if h.get("content")]
    lines = "\n".join(
        f"{'使用者' if h['role'] == 'user' else '助理'}：{h['content'][:150]}"
        for h in recent
    )
    prompt = f"""從以下旅遊對話中，提取旅客的旅遊偏好，回傳 JSON（欄位找不到資訊時填 null）：

對話：
{lines}

回傳格式（純 JSON，不可包含其他文字）：
{{
  "trip_type": "根據對話內容用2-6個中文字描述旅遊偏好風格，例如：自然生態、文化宗教、冒險健行、輕鬆休閒、親子友善、銀髮輕旅遊 等，或 null",
  "companion": "個人獨旅 或 情侶同遊 或 親子同遊 或 長輩同行 或 null",
  "travel_month": "X月 格式（如 10月）或 null",
  "days": "整數天數（如 5），若對話中未提及則填 null"
}}"""
    try:
        raw = gemini_call(prompt, json_mode=True)
        return json.loads(raw)
    except Exception:
        return {}


# ============================================================
# 9️⃣ Session state 初始化
# ============================================================
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'trend_counter' not in st.session_state:
    st.session_state.trend_counter = Counter()
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
if 'extracted_params' not in st.session_state:
    st.session_state.extracted_params = {}
if 'itinerary_month_num' not in st.session_state:
    st.session_state.itinerary_month_num = 0

def log_trend(emotion_label):
    st.session_state.emotion_log.append(emotion_label)
    st.session_state.trend_counter = Counter(st.session_state.emotion_log)
    return st.session_state.trend_counter


# ============================================================
# 🔟 主畫面：Claude.ai 風格雙欄版面
# ============================================================
col_left, col_right = st.columns(2, gap="medium")

# 計算可用高度：注入 JS 把視窗高度寫入 hidden input，Streamlit 無法直接讀取
# 改用固定比例：總可用高度 = vh - 頂部固定元素
# 左欄：header(~70px) + divider(16px) + textarea(90px) + submit(40px) + padding(20px) = ~236px overhead
# 右欄：toolbar(~50px) = overhead
# 兩側 history/card 容器高度設為 calc(100vh - overhead)
# Streamlit container(height=) 只接受 px，用估算值
import streamlit.components.v1 as components

# 注入 JS：動態設 session_state.viewport_h（只在第一次載入執行）
if 'viewport_h' not in st.session_state:
    st.session_state.viewport_h = 700   # 預設值

components.html("""
<script>
const h = window.innerHeight || document.documentElement.clientHeight;
const query = new URLSearchParams(window.location.search);
// 透過 URL fragment 傳高度（Streamlit 無法直接接收 JS postMessage）
// 實際做法：用固定視窗高度估算
</script>
""", height=0)

# 用視窗高度估算各容器高度
# 左欄 overhead = title(60) + caption(20) + divider(20) + textarea(90) + submit_btn(42) + padding(30) ≈ 262
# 右欄 overhead = toolbar(50) + padding(20) ≈ 70
_LEFT_OVERHEAD  = 262
_RIGHT_OVERHEAD = 70
_BASE_VH        = 780   # 一般 1080p 視窗扣掉瀏覽器工具列的保守估計

HISTORY_H  = 360
CARD_H     = 360 + (_LEFT_OVERHEAD - _RIGHT_OVERHEAD)  # 360 + (262-70) = 552

# ══════════════════════════════════════════════════════════════
# 左欄：對話區
# 上：st.container(height=) 捲動歷史  /  下：st.form 輸入框
# ══════════════════════════════════════════════════════════════
with col_left:
    st.markdown(
        "### 🏔️ 尼泊爾旅遊設計師\n"
        "<p style='font-size:12px;color:#888;margin:-8px 0 8px 0;'>"
        "與 AI 旅伴聊聊，說「幫我規劃行程」即可生成行程</p>",
        unsafe_allow_html=True,
    )

    # ── 對話歷史：原生捲動容器 ──────────────────────────────
    history_box = st.container(height=HISTORY_H, border=False)
    with history_box:
        if not st.session_state.conversation_history:
            st.markdown(
                f"<div style='display:flex;flex-direction:column;align-items:center;"
                f"justify-content:center;height:{HISTORY_H - 20}px;text-align:center;"
                f"color:#bbb;font-size:13px;'>"
                f"🌏 還沒有對話紀錄<br>在下方輸入框開始聊聊吧！</div>",
                unsafe_allow_html=True,
            )
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ── 輸入框：st.form 確保在欄位底部 ─────────────────────
    st.divider()
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            label="",
            placeholder="詢問旅遊建議，或說「幫我規劃行程」…\n（Shift+Enter 換行，按送出送出）",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("送出 ➤", use_container_width=True)

    query = user_input.strip() if submitted and user_input.strip() else None

    if query:
        if not wants_itinerary(query):
            st.session_state.show_confirmation = False

        st.session_state.conversation_history.append({
            "role": "user", "content": query, "query": query
        })

        with st.spinner("AI 旅伴正在思考..."):
            try:
                docs = retrieve(query)
                answer = generate_answer(query, docs, st.session_state.conversation_history)
                emotion_label, score = analyze_emotion(answer)
                trend = log_trend(emotion_label)
                source_names = docs['景點名稱_中文'].tolist()

                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "answer": answer,
                    "emotion_label": emotion_label,
                    "emotion_score": score,
                    "source": source_names,
                    "trend_snapshot": dict(trend),
                })

            except Exception as e:
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": f"ChatBot 發生錯誤：{e}",
                    "error": str(e),
                })

        if wants_itinerary(query):
            params = extract_trip_params(st.session_state.conversation_history)
            st.session_state.extracted_params = params
            st.session_state.show_confirmation = True

        st.rerun()

# ══════════════════════════════════════════════════════════════
# 右欄：行程面板（捲動）
# ══════════════════════════════════════════════════════════════
with col_right:
    # ── 頂部工具列 ────────────────────────────────────────────
    rc1, rc2, rc3 = st.columns([3, 1.2, 1])
    with rc1:
        planned = len(st.session_state.itinerary['days']) if st.session_state.itinerary else 0
        label = f"📅 已規劃 {planned} 天行程" if planned else "📋 行程面板"
        st.markdown(f"<p style='font-size:15px;font-weight:600;color:#1a1a1a;margin:6px 0;'>{label}</p>", unsafe_allow_html=True)
    with rc2:
        if st.session_state.itinerary:
            export_params = {**st.session_state.extracted_params}
            # 補上使用者在確認小卡實際選的值（存在 session_state）
            for k, sk in [('companion','confirm_companion'),('trip_type','confirm_trip_type')]:
                if sk in st.session_state:
                    export_params[k] = st.session_state[sk]
            html_bytes = build_export_html(
                st.session_state.itinerary,
                export_params,
                st.session_state.itinerary_month_num,
            ).encode('utf-8')
            st.download_button(
                label="📥 匯出行程",
                data=html_bytes,
                file_name="nepal_itinerary.html",
                mime="text/html",
                use_container_width=True,
            )
    with rc3:
        if st.button("🔄 清空", use_container_width=True):
            st.session_state.itinerary = None
            st.session_state.conversation_history = []
            st.session_state.emotion_log = []
            st.session_state.trend_counter = Counter()
            st.session_state.show_confirmation = False
            st.session_state.extracted_params = {}
            st.session_state.itinerary_month_num = 0
            st.rerun()

    # ── 喜好確認小卡 ──────────────────────────────────────────
    if st.session_state.show_confirmation:
        params = st.session_state.extracted_params

        # 天數：優先用 AI 萃取的 days，次用 regex fallback
        ai_days = None
        raw_days = params.get('days')
        if raw_days is not None:
            try:
                ai_days = min(14, max(1, int(str(raw_days).replace('天', '').strip())))
            except (ValueError, TypeError):
                ai_days = None
        if ai_days is None:
            for h in reversed(st.session_state.conversation_history[-12:]):
                m = re.search(r'(\d+)\s*天', h.get('content', ''))
                if m:
                    ai_days = min(14, max(1, int(m.group(1))))
                    break
        ai_days = ai_days or 3

        # 旅伴
        companion_options = ["個人獨旅", "情侶同遊", "親子同遊", "長輩同行"]
        ai_companion = params.get('companion') or '個人獨旅'
        ai_companion_idx = companion_options.index(ai_companion) if ai_companion in companion_options else 0

        # 旅遊偏好（自由文字，AI 預填）
        ai_trip_type = params.get('trip_type') or ''

        # 月份
        ai_month = params.get('travel_month') or '未指定'

        with st.container(border=True):
            st.markdown("### 🗺️ 已掌握您的旅程輪廓")

            # AI 判讀摘要列
            tags = []
            if ai_companion != '個人獨旅' or params.get('companion'):
                tags.append(f"👥 {ai_companion}")
            tags.append(f"🧭 {ai_trip_type}")
            if ai_month != '未指定':
                tags.append(f"🗓️ {ai_month}出發")
            tags.append(f"⏱️ {ai_days} 天")
            st.markdown(
                " &nbsp;｜&nbsp; ".join(
                    f'<span style="background:#e8f0e8;color:#3d5a3d;padding:2px 8px;border-radius:10px;font-size:12px;">{t}</span>'
                    for t in tags
                ),
                unsafe_allow_html=True,
            )
            st.caption("AI 已根據對話自動預填，可直接調整後生成行程。")
            st.divider()

            confirm_days = st.number_input(
                "行程天數", min_value=1, max_value=14,
                value=ai_days, step=1, key="confirm_days"
            )
            confirm_companion = st.selectbox(
                "旅伴類型", companion_options,
                index=ai_companion_idx, key="confirm_companion"
            )
            confirm_trip_type = st.text_input(
                "旅遊偏好（AI 已根據對話預填，可直接修改）",
                value=ai_trip_type,
                placeholder="例如：自然生態、文化宗教、冒險健行、輕鬆休閒…",
                key="confirm_trip_type"
            )
            confirm_must = st.multiselect(
                "指定景點（選填）", df['景點名稱_中文'].tolist(), key="confirm_must"
            )

            g1, g2 = st.columns([2, 1])
            with g1:
                if st.button("✨ 為我生成專屬行程", type="primary", use_container_width=True):
                    final_month = params.get('travel_month') or ''
                    month_num = int(final_month.replace("月", "")) if final_month and '月' in final_month else 0
                    with st.spinner("✈️ 旅伴正在為您打包行程…"):
                        try:
                            result = generate_itinerary(
                                confirm_trip_type, int(confirm_days), final_month,
                                confirm_companion,
                                st.session_state.conversation_history,
                                confirm_must,
                            )
                            st.session_state.itinerary = result
                            st.session_state.itinerary_month_num = month_num
                            st.session_state.show_confirmation = False

                            # ── 行程生成後同步摘要到左側對話歷史 ──
                            day_lines = "\n".join(
                                "・".join(s['name'] for s in d['spots'])
                                for d in result['days']
                            )
                            summary = (
                                f"✅ 已為你生成 **{int(confirm_days)} 天{confirm_trip_type}行程**"
                                f"（{confirm_companion}，{final_month or '月份未定'}出發）\n\n"
                                + "\n\n".join(
                                    f"**第 {d['day']} 天**：{'、'.join(s['name'] for s in d['spots'])}"
                                    for d in result['days']
                                )
                                + "\n\n詳細小卡請見右側行程面板 👉"
                            )
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": summary,
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"行程生成失敗：{e}")
            with g2:
                if st.button("繼續聊聊", use_container_width=True):
                    st.session_state.show_confirmation = False
                    st.rerun()

    # ── 行程卡片（獨立捲動容器）─────────────────────────────
    if st.session_state.itinerary:
        with st.container(height=CARD_H, border=False):
            show_itinerary_cards(st.session_state.itinerary, st.session_state.itinerary_month_num)

    # ── 尚無行程時的引導畫面 ──────────────────────────────────
    if not st.session_state.itinerary and not st.session_state.show_confirmation:
        st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
height:60vh;text-align:center;color:#999;">
  <div style="font-size:48px;margin-bottom:16px;">🏔️</div>
  <p style="font-size:16px;font-weight:500;color:#555;margin:0 0 8px 0;">行程面板</p>
  <p style="font-size:13px;color:#aaa;margin:0;">在左側與 AI 旅伴聊聊你的偏好<br>說「幫我規劃行程」即可在此生成專屬行程</p>
</div>
""", unsafe_allow_html=True)
