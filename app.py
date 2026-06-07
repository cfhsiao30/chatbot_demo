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
</style>
""", unsafe_allow_html=True)

st.title("🏔️ 尼泊爾旅遊設計師")
st.caption("打造專屬你的夢幻尼泊爾之旅")
st.info("🎒 旅途準備好了嗎？AI旅伴已打包好地圖與好心情。")

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
# 4️⃣ 景點圖片（Wikimedia Commons，CC BY-SA 授權）
# ============================================================
# 本地 images/ 資料夾優先；找不到時使用以下 Wikimedia 備援圖
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

def load_spot_image(name: str):
    """優先用本地 images/ 資料夾；找不到則用 Wikimedia Commons 備援圖。
    回傳 (來源, 圖片路徑或URL)，來源為 'local' 或 'wikimedia'。"""
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        path = Path(f"images/{name}{ext}")
        if path.exists():
            return 'local', str(path)
    if name in WIKIMEDIA_IMAGES:
        return 'wikimedia', WIKIMEDIA_IMAGES[name]
    return None, None

# ============================================================
# 5️⃣ 行程生成（Gemini）
# ============================================================
TRIP_TYPE_MAP = {
    "冒險健行": ["安娜普納基地營", "聖母峰基地營健行", "朗塘國家公園"],
    "文化宗教": ["斯瓦揚布佛塔", "帕舒帕提那寺", "藍毗尼", "巴克塔普爾杜巴廣場"],
    "自然生態": ["博卡拉", "奇特旺國家公園", "巴迪亞國家公園"],
}

def generate_itinerary(trip_type: str, total_days: int) -> dict:
    priority_spots = TRIP_TYPE_MAP.get(trip_type, [])
    all_spots_info = "\n".join([
        f"- {row['景點名稱_中文']}（{row['地點類型']}，停留：{row['平均停留時間建議']}）"
        for _, row in df.iterrows()
    ])
    priority_str = "、".join(priority_spots)

    prompt = f"""你是尼泊爾旅遊專家。請為偏好「{trip_type}」的旅客規劃 {total_days} 天行程。

優先景點（請優先從以下選擇，景點名稱必須完全一致）：{priority_str}

所有可選景點：
{all_spots_info}

請只回傳純 JSON，不要任何說明文字或 markdown 標記：
{{
  "days": [
    {{
      "day": 1,
      "spots": [
        {{
          "name": "景點中文名稱",
          "duration": "建議停留時間",
          "food": "餐食建議（一句話）",
          "tip": "小提醒（一句話）"
        }}
      ]
    }}
  ]
}}

每天安排 2–3 個景點，依旅客偏好選擇最合適的景點。"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return json.loads(response.text)

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


def show_spot_card(spot: dict):
    name = spot['name']
    source, img = load_spot_image(name)

    # 外框：米白底 + 左邊綠線
    with st.container():
        st.markdown('<div style="background:#F5F0E8;border-left:4px solid #5B8A5B;border-radius:10px;padding:12px;margin-bottom:12px;">', unsafe_allow_html=True)

        img_col, info_col = st.columns([1, 1.8])

        with img_col:
            if img:
                src = _img_src(source, img)
                # padding-top:100% 技巧讓圖片強制正方形
                st.markdown(f'<div style="position:relative;width:100%;padding-top:100%;overflow:hidden;border-radius:8px;"><img src="{src}" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;"></div>', unsafe_allow_html=True)
                if source == 'wikimedia':
                    st.caption("© Wikimedia Commons CC BY-SA")
            else:
                st.markdown('<div style="position:relative;width:100%;padding-top:100%;background:#D4CFC7;border-radius:8px;"><span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#6B6B6B;font-size:13px;">📷 示意圖</span></div>', unsafe_allow_html=True)

        with info_col:
            st.markdown(f"**📍 {name}**")
            st.markdown(f"⏱️ {spot['duration']}")
            st.markdown(f"🍽️ {spot['food']}")
            st.markdown(f'<p style="color:#E07B39;margin:0;">💡 {spot["tip"]}</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def show_itinerary_cards(itinerary_data: dict):
    for day_data in itinerary_data['days']:
        day_num = day_data['day']
        st.markdown(f"### 📅 第 {day_num} 天行程")
        for spot in day_data['spots']:
            show_spot_card(spot)


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

# ============================================================
# 🔟 側欄
# ============================================================
with st.sidebar:
    st.markdown("### 🧭 旅遊偏好")
    trip_type = st.selectbox(
        "選擇您偏好的旅遊類型",
        ["冒險健行", "文化宗教", "自然生態"],
    )

    st.markdown("### 📅 規劃進度")
    total_days = st.number_input("請輸入行程總天數", min_value=1, max_value=14, value=3, step=1)

    planned = len(st.session_state.itinerary['days']) if st.session_state.itinerary else 0
    st.write(f"已完成 {planned}/{total_days} 天")
    st.progress(min(planned / total_days, 1.0) if total_days > 0 else 0)

    st.markdown("### 🗺️ 參考行程")
    if st.button("生成建議行程", use_container_width=True, type="primary"):
        with st.spinner("AI 正在規劃行程..."):
            try:
                result = generate_itinerary(trip_type, total_days)
                st.session_state.itinerary = result
                st.rerun()
            except Exception as e:
                st.error(f"行程生成失敗：{e}")

# ============================================================
# 主畫面：行程卡片 → 地圖 → 聊天
# ============================================================
if st.session_state.itinerary:
    show_itinerary_cards(st.session_state.itinerary)
    show_map(st.session_state.itinerary)
    st.markdown("---")

st.markdown("### 💬 旅遊問答")

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def log_trend(emotion_label):
    st.session_state.emotion_log.append(emotion_label)
    st.session_state.trend_counter = Counter(st.session_state.emotion_log)
    return st.session_state.trend_counter


if query := st.chat_input("輸入你的旅遊問題..."):
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
