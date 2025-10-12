import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from collections import Counter
import os
from google import genai

# Set Streamlit page config
st.set_page_config(layout="wide")

st.write("#### 🌐 尼泊爾旅遊規劃夥伴")

# ============================================================
# 1️⃣ 資料載入（模擬 RAG 資料庫）
# ============================================================
# Use st.cache_resource to cache the data loading and processing
@st.cache_resource
def load_data():
    df = pd.DataFrame([
        {"景點名稱":"Syambhunath","景點名稱_中文":"斯瓦揚布佛塔（Syambhunath）","好評度":"94%","評價比例":"878/35/17/0","智慧摘要":"強烈推薦","lat":27.7148996,"lng":85.2903957,"types":["tourist_attraction","place_of_worship","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；天氣涼爽","特色活動或體驗":["猴廟參觀","盆地全景觀賞","佛教文化體驗","觀賞加德滿都全景","參拜佛塔","與猴群互動"],"回答方向提示":"可形容為「加德滿都的制高點，景色壯觀、猴子可愛」","地點類型":"宗教聖地","適合族群建議":["情侶","攝影愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["人潮擁擠","猴子搶食物"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Annapurna Base Camp","景點名稱_中文":"安娜普納基地營（Annapurna Base Camp）","好評度":"87%","評價比例":"115/5/12/0","智慧摘要":"可安排短暫造訪","lat":28.53,"lng":83.878,"types":["natural_feature","establishment"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["生態區域穿越","高山景觀觀賞","健行","高山健行","露營","欣賞雪山日出"],"回答方向提示":"可描述「經典登山路線」，提醒注意體能與天氣","地點類型":"冒險健行","適合族群建議":["背包客","冒險愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["高海拔引起高山症","天氣變化快"],"關聯景點建議":["Pokhara","Ghorepani Poon Hill"]},
        {"景點名稱":"Bardiya National Park","景點名稱_中文":"巴迪亞國家公園（Bardiya National Park）","好評度":"91%","評價比例":"541/29/25/0","智慧摘要":"值得一遊","lat":28.3648644,"lng":81.5596427,"types":["tourist_attraction","park","point_of_interest","establishment"],"最佳造訪季節":"11-2月；乾季","特色活動或體驗":["叢林野生動物觀察","孟加拉虎與犀牛觀察","叢林探險","野生動物觀察","叢林徒步","搭乘獨木舟"],"回答方向提示":"可提「安靜且原始的國家公園」，適合熱愛野生自然的旅人","地點類型":"自然風景","適合族群建議":["野生動物愛好者","家庭"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["交通不便","導遊服務質量參差"],"關聯景點建議":["Chitwan National Park"]},
        {"景點名稱":"Bhaktapur Durbar Square","景點名稱_中文":"巴克塔普爾杜巴廣場（Bhaktapur Durbar Square）","好評度":"94%","評價比例":"874/29/27/0","智慧摘要":"強烈推薦","lat":27.67207,"lng":85.4282951,"types":["tourist_attraction","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["中世紀建築參觀","雕刻藝術欣賞","手工藝文化體驗","探索古老建築","品嚐當地美食","參觀傳統工藝坊"],"回答方向提示":"可強調「古城風情與文化遺產」，適合愛歷史或攝影的人","地點類型":"文化古蹟","適合族群建議":["歷史愛好者","家庭"],"平均停留時間建議":"2–3 小時","常見抱怨或負評原因":["建築受損","重建工程中"],"關聯景點建議":["Kathmandu Durbar Square","Patan Durbar Square"]},
        {"景點名稱":"Chitwan National Park","景點名稱_中文":"奇特旺國家公園（Chitwan National Park）","好評度":"91%","評價比例":"844/48/38/0","智慧摘要":"值得一遊","lat":27.519285,"lng":84.3135318,"types":["tourist_attraction","park","point_of_interest","establishment"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["騎象觀察動物","叢林探險","獨木舟觀察鳥類","野生動物觀察","叢林騎象","搭乘獨木舟"],"回答方向提示":"可描述為「適合體驗野生動物與自然生態的地方」","地點類型":"自然風景","適合族群建議":["家庭","野生動物愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["觀察野生動物機會不穩定","價格偏高"],"關聯景點建議":["Bardiya National Park"]},
        {"景點名稱":"Everest Base Camp Trek","景點名稱_中文":"聖母峰基地營健行（Everest Base Camp Trek）","好評度":"85%","評價比例":"1994/3/14/0","智慧摘要":"口碑普通 建議斟酌或查更多資訊","lat":28.0018515,"lng":86.8513119,"types":["tourist_attraction","travel_agency","park","point_of_interest","establishment"],"最佳造訪季節":"3-5月、9-11月；冬季極寒","特色活動或體驗":["高山健行","雪山觀景","高海拔挑戰","欣賞冰河與雪山景觀","體驗高山文化"],"回答方向提示":"可提「挑戰極限的經典健行路線」，但非所有人都適合","地點類型":"冒險健行","適合族群建議":["冒險愛好者","健行者"],"平均停留時間建議":"12–14 天","常見抱怨或負評原因":["行程艱辛","住宿條件簡陋"],"關聯景點建議":["Lukla","Namche Bazaar"]},
        {"景點名稱":"Langtang","景點名稱_中文":"朗塘國家公園（Langtang）","好評度":"92%","評價比例":"622/27/24/0","智慧摘要":"強烈推薦","lat":28.2062873,"lng":85.6229296,"types":["locality","political"],"最佳造訪季節":"3-5月、9-11月；氣候宜人","特色活動或體驗":["國家公園健行","藏族文化村落","冰川觀景","健行","觀山日出","村落住宿體驗","溫泉放鬆"],"回答方向提示":"可強調「景色壯麗、適合初中級登山者」","地點類型":"冒險健行","適合族群建議":["健行者","自然愛好者"],"平均停留時間建議":"7–10 天","常見抱怨或負評原因":["交通不便","住宿選擇有限"],"關聯景點建議":["Kathmandu","Gosaikunda Lake"]},
        {"景點名稱":"Lumbini","景點名稱_中文":"藍毗尼（Lumbini）","好評度":"94%","評價比例":"878/28/29/0","智慧摘要":"強烈推薦","lat":27.9207402,"lng":82.7347142,"types":["administrative_area_level_1","political"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["佛教聖地參觀","寺廟探索","冥想靜修","參訪佛教聖地","瞻仰佛塔","冥想與靜修"],"回答方向提示":"可強調「心靈沉澱、宗教意義深厚」的朝聖感","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1–2 小時","常見抱怨或負評原因":["設施簡陋","周邊環境需改善"],"關聯景點建議":["Tansen","Palpa"]},
        {"景點名稱":"Pasupatinath Temple","景點名稱_中文":"帕舒帕提那寺（Pasupatinath Temple）","好評度":"91%","評價比例":"849/33/48/0","智慧摘要":"強烈推薦","lat":27.710512,"lng":85.3488125,"types":["hindu_temple","place_of_worship","point_of_interest","establishment"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["印度教聖地參拜","觀賞火葬儀式","宗教文化體驗","參拜印度教聖地","觀賞火葬儀式","體驗宗教文化"],"回答方向提示":"可提「印度教最重要的聖地之一」，但提醒人潮較多","地點類型":"宗教聖地","適合族群建議":["宗教朝聖者","文化愛好者"],"平均停留時間建議":"1 小時","常見抱怨或負評原因":["人潮擁擠","香火濃烈"],"關聯景點建議":["Boudhanath Stupa","Pashupatinath Temple"]},
        {"景點名稱":"Pokhara","景點名稱_中文":"博卡拉（Pokhara）","好評度":"94%","評價比例":"880/45/11/0","智慧摘要":"強烈推薦","lat":28.2095831,"lng":83.9855674,"types":["locality","political"],"最佳造訪季節":"10–3 月；乾季","特色活動或體驗":["泛舟費瓦湖","安納普爾納日出","滑翔傘","洞窟與瀑布探索","湖上划船","瀑布探訪","滑翔傘","溫泉放鬆"],"回答方向提示":"可描述成「湖光山色、氣氛悠閒的城市」，適合放鬆與拍照","地點類型":"自然風景","適合族群建議":["情侶","家庭","攝影愛好者"],"平均停留時間建議":"2–3 天","常見抱怨或負評原因":["部分地區交通擁堵","設施需升級"],"關聯景點建議":["Sarangkot","Phewa Lake"]},
    ])

    useful_columns = [
        '好評度',
        '最佳造訪季節',
        '特色活動或體驗',
        '地點類型',
        '適合族群建議',
        '平均停留時間建議',
        '常見抱怨或負評原因',
        '關聯景點建議'
    ]

    def combine_info(row):
        info = f"智慧摘要: {row['智慧摘要']}。"
        if '好評度' in useful_columns:
            info += f" 好評度: {row['好評度']}。"
        if '最佳造訪季節' in useful_columns:
            info += f" 最佳造訪季節: {row['最佳造訪季節']}."
        if '特色活動或體驗' in useful_columns and isinstance(row['特色活動或體驗'], list):
            info += f" 特色活動或體驗: {', '.join(row['特色活動或體驗'])}。"
        if '地點類型' in useful_columns:
            info += f" 地點類型: {row['地點類型']}."
        if '適合族群建議' in useful_columns and isinstance(row['適合族群建議'], list):
            info += f" 適合族群建議: {', '.join(row['適合族群建議'])}."
        if '平均停留時間建議' in useful_columns:
            info += f" 平均停留時間建議: {row['平均停留時間建議']}."
        if '常見抱怨或負評原因' in useful_columns and isinstance(row['常見抱怨或負評原因'], list):
            info += f" 常見抱怨或負評原因: {', '.join(row['常見抱怨或負評原因'])}。"
        if '關聯景點建議' in useful_columns and isinstance(row['關聯景點建議'], list):
            info += f" 關聯景點建議: {', '.join(row['關聯景點建議'])}."
        return info

    df['combined_info'] = df.apply(combine_info, axis=1)
    return df

df = load_data()

# ============================================================
# 2️⃣ 建立向量資料庫（FAISS）
# ============================================================
# Use st.cache_resource to cache the model loading and index creation
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
# 3️⃣ 情緒細分類模型
# ============================================================
# Use st.cache_resource to cache the emotion analysis pipeline loading
@st.cache_resource
def load_emotion_analyzer():
    emotion_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512,
    )
    return emotion_analyzer

emotion_analyzer = load_emotion_analyzer()

def analyze_emotion(text):
    """分析文字情緒，回傳標籤與信心值"""
    result = emotion_analyzer(text[:2000])[0]
    return result['label'], result['score']

# ============================================================
# 4️⃣ 定義 RAG 檢索與回答生成
# ============================================================
def retrieve(query, top_k=2):
    """用 FAISS 搜尋最相關的景點資料"""
    query_vector = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    results = df.iloc[I[0]]
    return results

# Initialize Gemini client (assuming API key in environment)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY environment variable not set. Please add it to your Streamlit Cloud secrets.")
    st.stop() # Stop the app if API key is missing

client = genai.Client(api_key=API_KEY)

def generate_answer(query, retrieved_docs, history):
    """使用 Gemini 整合 RAG + 多輪對話記憶"""
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
"""
         for _, row in retrieved_docs.iterrows()]
    )

    # Ensure history contains dictionaries with 'query' and 'answer' keys
    memory = "\n".join([f"使用者：{h.get('query', '')}\n助理：{h.get('answer', '')}" for h in history[-5:]])  # 保留最近5輪, Use get for safety


    # 加入情緒回饋 (從歷史記錄中獲取最後一輪的情緒)
    emotion_feedback = ""
    if history:
        last_turn = history[-1]
        # Ensure last_turn is a dictionary and has the expected keys
        if isinstance(last_turn, dict):
            emotion_label = last_turn.get('emotion_label', '未知') # Use .get() for safety
            score = last_turn.get('emotion_score', 0.0) # Use .get() for safety
            emotion_feedback = f"請注意，上一次回答的情緒是 {emotion_label} (信心值 {score:.3f})，請在本次回答中保持親切、專業的語氣，並依據對話歷史來調整回應風格。"


    prompt = f"""
你是一位專業的尼泊爾旅遊助理，請參考以下提供的景點資訊來回答使用者的問題。
請務必**全面整合**檢索到的景點資訊，包含「好評度」、「最佳造訪季節」、「特色活動或體驗」、「地點類型」、「適合族群建議」、「平均停留時間建議」、「常見抱怨或負評原因」和「關聯景點建議」，提供**更全面且個人化**的旅遊建議。
如果使用者提到具體的偏好（例如：人少、交通便利、特定活動、逛市場等），請優先考量並在回答中**明確回應**這些需求。
在推薦景點時，可以簡要提及不同景點之間的**地理位置或交通**考量，幫助使用者規劃順遊路線。
請以親切、專業、**自然的語氣**，用中文、在**5句話內**生成回答，並推薦合適景點。
回覆請儘量避免重複、太過於制式，並結合對話歷史來維持連貫性。
補充說明：好評度>91%，表示最值得推薦，若是低於91%則不用強調好評度。

以下是使用者與助理的對話歷史：
{memory}

{emotion_feedback}

以下是檢索到的尼泊爾景點相關資訊：
{context}

使用者問題：{query}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

# ============================================================
# 5️⃣ Streamlit 介面與對話管理
# ============================================================

# Initialize chat history and trend in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'trend_counter' not in st.session_state:
    st.session_state.trend_counter = Counter()

def log_trend(emotion_label):
    st.session_state.emotion_log.append(emotion_label)
    st.session_state.trend_counter = Counter(st.session_state.emotion_log)
    return st.session_state.trend_counter

# Display chat history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display emotion and source only for assistant messages
        if message["role"] == "assistant":
             if "emotion_label" in message:
                  st.caption(f"🧠 回覆情緒： {message['emotion_label']}（信心值 {message['emotion_score']:.3f}）")
             if "source" in message:
                  st.caption(f"📌 引用資料來源：{', '.join(message['source'])}")


# Chat input
if query := st.chat_input("輸入你的旅遊問題..."):
    # Add user message to chat history
    st.session_state.conversation_history.append({"role": "user", "content": query, "query": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Process the query and generate response
    with st.chat_message("assistant"):
        with st.spinner("ChatBot 正在思考..."):
            try:
                docs = retrieve(query)
                answer = generate_answer(query, docs, st.session_state.conversation_history)
                emotion_label, score = analyze_emotion(answer)
                trend = log_trend(emotion_label) # Update trend counter
                source_names = docs['景點名稱_中文'].tolist()

                # Add assistant message to chat history, including metadata for history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "answer": answer, # Store answer for history in generate_answer
                    "emotion_label": emotion_label,
                    "emotion_score": score,
                    "source": source_names,
                    "trend_snapshot": dict(trend) # Optional: store trend snapshot per turn
                })

                # Display assistant message and metadata
                st.markdown(answer)
              # st.caption(f"🧠 回覆情緒： {emotion_label}（信心值 {score:.3f}）")
              # st.caption(f"📌 引用資料來源：{', '.join(source_names)}")


            except Exception as e:
                st.error(f"ChatBot 發生錯誤：{e}")
                # Log error in history if needed
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": f"ChatBot 發生錯誤：{e}",
                    "error": str(e)
                })


# Optional: Display trend counter in a sidebar or expansion
# with st.sidebar:
#     st.header("情緒趨勢統計")
#     if st.session_state.trend_counter:
#         # Convert Counter to dictionary for st.bar_chart
#         st.bar_chart(dict(st.session_state.trend_counter))
#     else:
#         st.info("暫無情緒數據")
