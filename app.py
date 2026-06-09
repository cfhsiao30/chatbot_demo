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

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(layout="wide", page_title="尼泊爾旅遊設計師")

st.markdown("""
<style>
.day-header {
    background: #5B8A5B;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 20px 0 10px 0;
}
.spot-card {
    background: #F5F0E8;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #5B8A5B;
}
/* Fix 5：進度條改綠色 */
div[data-testid="stProgressBar"] > div > div > div {
    background-color: #5B8A5B !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏔️ 尼泊爾旅遊設計師")
st.caption("打造專屬你的夢幻尼泊爾之旅｜先和旅伴聊聊，說「幫我規劃行程」即可生成專屬行程")

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
def fetch_commons_image(search_query: str, fallback_parent: str = "") -> str | None:
    """以 search_query 精準搜尋 Wikimedia Commons 照片。
    找不到真實照片時，以 fallback_parent 英文名再試一次。"""
    for query in dict.fromkeys([search_query, WIKIPEDIA_TITLES.get(fallback_parent, "")]):
        if not query:
            continue
        try:
            params = {
                "action": "query",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": query,
                "gsrlimit": "10",
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
                    if url and _is_photo_url(url):
                        return url
        except Exception:
            continue
    return None


def load_spot_image(search_query: str = "", parent: str = ""):
    """圖片來源優先順序：
    ① Wikimedia Commons（以 search_query 精準搜）
    ② 本地 images/（開發者覆蓋：放同名圖片即可替換）
    ③ Wikimedia 硬編碼備援
    回傳 (來源, 路徑或URL)，來源為 'web' / 'local' / 'wikimedia'。"""
    commons_url = fetch_commons_image(search_query, parent)
    if commons_url:
        return 'web', commons_url
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        path = Path(f"images/{parent}{ext}")
        if path.exists():
            return 'local', str(path)
    if parent in WIKIMEDIA_IMAGES:
        return 'wikimedia', WIKIMEDIA_IMAGES[parent]
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    result = json.loads(response.text)
    # 強制對齊天數：截斷多餘或補齊不足的天數
    days = result.get("days", [])
    if len(days) > total_days:
        result["days"] = days[:total_days]
    elif len(days) < total_days:
        for i in range(len(days) + 1, total_days + 1):
            result["days"].append({
                "day": i,
                "spots": [{"name": f"第{i}天自由活動", "parent": "博卡拉",
                            "search_query": "Pokhara Nepal", "duration": "彈性安排",
                            "food": "依個人喜好", "tip": "可依當天狀況調整"}]
            })
    # 重新編號確保連續
    for idx, d in enumerate(result["days"]):
        d["day"] = idx + 1
    return result

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

 **情緒即時回饋**
 - 參考使用者的情緒或回饋（{emotion_feedback}），調整語氣和建議內容。

以下是對話歷史：
{memory}

檢索到的景點資訊：
{context}

使用者問題：
{query}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

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


def _spot_card_html(spot: dict, travel_month_num: int = 0, used_urls: set = None) -> str:
    """
    橫式卡片：左圖（1/3）右文（2/3），主標在整張卡片底部。
    used_urls：同一天已使用的圖片 URL set，避免重複。
    """
    if used_urls is None:
        used_urls = set()

    name = spot['name']
    parent = spot.get('parent', name)
    search_query = spot.get('search_query', name)

    # 優先用精準 search_query 搜圖；若結果已被同天其他卡佔用則 fallback 到 parent 備援
    source, img = load_spot_image(search_query, parent)
    if img and img in used_urls:
        # 嘗試用 parent 換一張不同圖
        alt_source, alt_img = load_spot_image(parent + " Nepal", parent)
        if alt_img and alt_img not in used_urls:
            source, img = alt_source, alt_img
        # 若還是重複，就用佔位符
        elif alt_img and alt_img in used_urls:
            img = None

    if img:
        used_urls.add(img)

    season_warning = False
    if travel_month_num:
        match = df[df['景點名稱_中文'] == parent]
        if not match.empty:
            season_warning = not is_best_season(match.iloc[0]['最佳造訪季節'], travel_month_num)

    # ── 圖片區塊 ──
    if img:
        src = _img_src(source, img)
        credit = (
            '<div style="font-size:9px;color:#aaa;padding:1px 4px;'
            'background:rgba(0,0,0,0.35);position:absolute;bottom:0;left:0;right:0;">'
            '© Wikipedia CC BY-SA</div>'
        ) if source in ('web', 'wikimedia') else ''
        img_block = (
            f'<div style="position:relative;width:120px;min-width:120px;height:100%;'
            f'background:#ccc;border-radius:10px 0 0 10px;overflow:hidden;">'
            f'<img src="{src}" style="width:120px;height:100%;object-fit:cover;display:block;">'
            f'{credit}'
            f'</div>'
        )
    else:
        img_block = (
            f'<div style="width:120px;min-width:120px;height:100%;background:#D4CFC7;'
            f'border-radius:10px 0 0 10px;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;color:#6B6B6B;font-size:11px;">📷</div>'
        )

    warning_html = (
        '<span style="color:#92650a;font-size:10px;background:#fff3cd;'
        'padding:2px 5px;border-radius:3px;display:inline-block;margin-top:4px;">⚠️ 非最佳造訪季節</span>'
    ) if season_warning else ''

    # ── 文字區塊（右側）──
    text_block = (
        f'<div style="flex:1;padding:10px 12px;display:flex;flex-direction:column;justify-content:center;min-width:0;">'
        f'<p style="margin:0 0 6px 0;font-weight:bold;color:#3D5A3D;font-size:13px;line-height:1.4;">📍 {name}</p>'
        f'<p style="margin:3px 0;color:#555;font-size:11px;">⏱️ {spot["duration"]}</p>'
        f'<p style="margin:3px 0;color:#555;font-size:11px;">🍽️ {spot["food"]}</p>'
        f'<p style="margin:3px 0;color:#E07B39;font-size:11px;">💡 {spot["tip"]}</p>'
        f'{warning_html}'
        f'</div>'
    )

    return (
        f'<div style="display:flex;flex-direction:row;width:100%;background:#F5F0E8;'
        f'border-radius:10px;border:1px solid #d8d0c4;overflow:hidden;min-height:110px;">'
        f'{img_block}'
        f'{text_block}'
        f'</div>'
    )


def show_itinerary_cards(itinerary_data: dict, travel_month_num: int = 0):
    for day_data in itinerary_data['days']:
        day_num = day_data['day']
        st.markdown(f"### 📅 第 {day_num} 天行程")
        used_urls: set = set()
        cards_html = "".join(
            _spot_card_html(spot, travel_month_num, used_urls)
            for spot in day_data['spots']
        )
        st.markdown(
            f'<div style="display:flex;flex-direction:column;gap:10px;padding:4px 2px 16px 2px;">'
            f'{cards_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 8️⃣ 互動地圖
# ============================================================
def show_map(itinerary_data: dict):
    st.markdown("### 🗺️ 行程地圖")

    if not HAS_FOLIUM:
        st.warning("請安裝地圖套件：`pip install folium streamlit-folium`")
        return

    all_spots = []
    for day_data in itinerary_data['days']:
        for spot in day_data['spots']:
            match = df[df['景點名稱_中文'] == spot['name']]
            if not match.empty:
                row = match.iloc[0]
                all_spots.append({
                    'name': spot['name'],
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'day': day_data['day'],
                })

    if not all_spots:
        st.info("找不到行程景點的地理座標。")
        return

    center_lat = sum(s['lat'] for s in all_spots) / len(all_spots)
    center_lng = sum(s['lng'] for s in all_spots) / len(all_spots)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=7)

    day_colors = ['green', 'blue', 'red', 'purple', 'orange', 'darkgreen', 'darkblue']
    for spot in all_spots:
        color = day_colors[(spot['day'] - 1) % len(day_colors)]
        folium.Marker(
            location=[spot['lat'], spot['lng']],
            popup=f"第 {spot['day']} 天：{spot['name']}",
            tooltip=spot['name'],
            icon=folium.Icon(color=color, icon='info-sign'),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=420)


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
  "trip_type": "冒險健行 或 文化宗教 或 自然生態 或 null",
  "companion": "個人獨旅 或 情侶同遊 或 親子同遊 或 長輩同行 或 null",
  "travel_month": "X月 格式（如 10月）或 null",
  "days": "整數天數（如旅客提到幾天、幾日遊等，提取數字）或 null"
}}"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
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

# ============================================================
# 🔟 側欄
# ============================================================
with st.sidebar:
    st.markdown("### 📅 規劃進度")
    planned = len(st.session_state.itinerary['days']) if st.session_state.itinerary else 0
    st.write(f"已規劃 {planned} 天行程")

    if st.button("🔄 清空重來", use_container_width=True):
        st.session_state.itinerary = None
        st.session_state.conversation_history = []
        st.session_state.emotion_log = []
        st.session_state.trend_counter = Counter()
        st.session_state.show_confirmation = False
        st.session_state.extracted_params = {}
        st.session_state.itinerary_month_num = 0
        st.rerun()

# ============================================================
# 主畫面：左欄=聊天，右欄=行程
# ============================================================
def log_trend(emotion_label):
    st.session_state.emotion_log.append(emotion_label)
    st.session_state.trend_counter = Counter(st.session_state.emotion_log)
    return st.session_state.trend_counter


col_chat, col_itinerary = st.columns([1, 1], gap="large")

# ──────────────── 左欄：聊天 ────────────────
with col_chat:
    st.markdown("### 💬 旅遊問答")

    # 對話歷史
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── 確認卡片：AI 偵測到規劃意圖後出現 ──
    if st.session_state.show_confirmation:
        params = st.session_state.extracted_params

        # AI 判讀結果作為預設值
        TRIP_TYPE_OPTIONS = ["冒險健行", "文化宗教", "自然生態"]
        COMPANION_OPTIONS = ["個人獨旅", "情侶同遊", "親子同遊", "長輩同行"]

        ai_trip_type = params.get('trip_type') or "文化宗教"
        ai_companion = params.get('companion') or "個人獨旅"
        ai_month     = params.get('travel_month') or ''

        # 解析 AI 判讀天數，預設 3
        ai_days_raw = params.get('days')
        try:
            ai_days = int(ai_days_raw) if ai_days_raw else 3
            ai_days = max(1, min(14, ai_days))
        except (ValueError, TypeError):
            ai_days = 3

        trip_type_idx = TRIP_TYPE_OPTIONS.index(ai_trip_type) if ai_trip_type in TRIP_TYPE_OPTIONS else 1
        companion_idx = COMPANION_OPTIONS.index(ai_companion) if ai_companion in COMPANION_OPTIONS else 0

        with st.container(border=True):
            st.markdown("### 🗺️ 我已掌握您的旅程輪廓！")
            st.caption("以下為 AI 根據對話判讀的偏好，您可直接調整後生成行程。")

            confirm_trip_type = st.selectbox(
                "旅遊偏好", TRIP_TYPE_OPTIONS, index=trip_type_idx, key="confirm_trip_type"
            )
            confirm_companion = st.selectbox(
                "旅伴類型", COMPANION_OPTIONS, index=companion_idx, key="confirm_companion"
            )
            confirm_month = st.text_input(
                "出發月份（如：10月）", value=ai_month, key="confirm_month"
            )
            confirm_days = st.number_input(
                "行程天數", min_value=1, max_value=14, value=ai_days, step=1, key="confirm_days"
            )
            confirm_must = st.multiselect(
                "指定景點（選填）", df['景點名稱_中文'].tolist(), key="confirm_must"
            )

            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                if st.button("✨ 為我生成專屬行程", type="primary", use_container_width=True):
                    final_month = confirm_month.strip()
                    month_num = int(final_month.replace("月", "")) if final_month and '月' in final_month else 0
                    with st.spinner("✈️ 旅伴正在為您打包行程…"):
                        try:
                            result = generate_itinerary(
                                confirm_trip_type, int(confirm_days), final_month, confirm_companion,
                                st.session_state.conversation_history,
                                confirm_must,
                            )
                            st.session_state.itinerary = result
                            st.session_state.itinerary_month_num = month_num
                            st.session_state.show_confirmation = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"行程生成失敗：{e}")
            with btn_col2:
                if st.button("繼續聊聊", use_container_width=True):
                    st.session_state.show_confirmation = False
                    st.rerun()

    # chat_input 必須在欄外才能固定底部（Streamlit 限制：放欄內也可運作）
    if query := st.chat_input("詢問旅遊建議，或說「幫我規劃行程」…"):
        if not wants_itinerary(query):
            st.session_state.show_confirmation = False

        st.session_state.conversation_history.append({
            "role": "user", "content": query, "query": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("ChatBot 正在思考..."):
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

                    st.markdown(answer)

                except Exception as e:
                    st.error(f"ChatBot 發生錯誤：{e}")
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

# ──────────────── 右欄：行程 ────────────────
with col_itinerary:
    if st.session_state.itinerary:
        st.markdown("### 🗓️ 您的專屬行程")
        show_itinerary_cards(st.session_state.itinerary, st.session_state.itinerary_month_num)
        show_map(st.session_state.itinerary)
    else:
        st.markdown("### 🗓️ 專屬行程")
        st.info("行程將在此顯示。先和旅伴聊聊，說「幫我規劃行程」即可生成！")
