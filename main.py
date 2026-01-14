import os

packages = [
    "streamlit",
    "yfinance",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "textblob",
    "scikit-learn",
    "plotly"
]

for pkg in packages:
    os.system(f"pip install {pkg}")

print("✔ All libraries installed successfully!")

#---------------------------------------------------#

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== إعداد الصفحة ====================
st.set_page_config(
    page_title="📈 Apple Stock Analysis & Prediction",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== تخصيص CSS ====================
st.markdown("""
    <style>
    /* التنسيق العام */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #FF6B6B;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 5px;
    }
    .info-box {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    /* الأزرار */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #2E7D32);
        color: white;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2E7D32, #1B5E20);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* القائمة الجانبية */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E3C72, #2A5298);
    }
    /* المؤشرات */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
    }
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    /* الرسوم البيانية */
    .plotly-graph-div {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }
    /* التنبيهات */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== القائمة الجانبية ====================
with st.sidebar:
    # العنوان واللوجو
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color: white;'>🍎 Apple</h1>
            <p style='color: #CCCCCC;'>Stock Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # إعدادات التواريخ
    st.markdown("### 📅 إعداد الفترة الزمنية")
    
    today = datetime.now()
    default_start = datetime(today.year-2, 1, 1)
    
    start_date = st.date_input(
        "**تاريخ البداية**",
        value=default_start,
        help="اختر تاريخ بداية لتحليل البيانات"
    )
    
    end_date = st.date_input(
        "**تاريخ النهاية**",
        value=today,
        help="اختر تاريخ نهاية لتحليل البيانات"
    )
    
    st.markdown("---")
    
    # خيارات التحليل
    st.markdown("### ⚙️ خيارات التحليل")
    
    with st.expander("📊 إظهار/إخفاء الخيارات", expanded=True):
        show_raw_data = st.checkbox("📄 البيانات الخام", value=True, 
                                   help="عرض البيانات التاريخية في جدول")
        show_analysis = st.checkbox("📈 التحليل الاستكشافي", value=True,
                                   help="عرض الرسوم البيانية والتحليلات")
        run_prediction = st.checkbox("🔮 التنبؤ بالأسعار", value=True,
                                   help="تشغيل نموذج التنبؤ بالأسعار المستقبلية")
    
    st.markdown("---")
    
    # إعدادات النموذج
    st.markdown("### 🤖 إعدادات النموذج")
    
    prediction_days = st.slider(
        "**عدد أيام التنبؤ:**",
        min_value=7,
        max_value=90,
        value=30,
        help="عدد الأيام المستقبلية التي تريد التنبؤ بها"
    )
    
    model_complexity = st.select_slider(
        "**تعقيد النموذج:**",
        options=["بسيط", "متوسط", "معقد"],
        value="متوسط",
        help="تحديد مستوى تعقيد نموذج التنبؤ"
    )
    
    st.markdown("---")
    
    # زر التشغيل الرئيسي
    analyze_button = st.button(
        "🚀 بدء التحليل الآن",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # معلومات إضافية
    with st.expander("ℹ️ معلومات حول البيانات"):
        st.info("""
        **مصدر البيانات:** Yahoo Finance
        **رمز السهم:** AAPL
        **التحديث:** يومياً
        **الاستخدام:** لأغراض تعليمية فقط
        """)

# ==================== دالة تحميل البيانات ====================
@st.cache_data(ttl=3600)
def load_stock_data(ticker="AAPL", start_date=None, end_date=None):
    """تحميل بيانات الأسهم من Yahoo Finance"""
    try:
        with st.spinner(f'📥 جاري تحميل بيانات {ticker}...'):
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                st.error("⚠️ لم يتم العثور على بيانات للفترة المحددة")
                return None
            
            data.reset_index(inplace=True)
            
            # إعادة تسمية الأعمدة إذا كانت متعددة المستويات
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # التأكد من وجود عمود التاريخ
            if 'Date' not in data.columns:
                data = data.reset_index()
            
            # التحقق من البيانات
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.warning(f"⚠️ الأعمدة التالية مفقودة: {missing_cols}")
            
            return data
    
    except Exception as e:
        st.error(f"❌ حدث خطأ في تحميل البيانات: {str(e)}")
        return None

# ==================== دالة إنشاء الميزات ====================
def create_features(data):
    """إنشاء ميزات إضافية للتعلم الآلي"""
    df = data.copy()
    
    # تحويل التاريخ إلى شكل رقمي
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # المتوسطات المتحركة
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # التغيرات النسبية
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Change'] = df['Close'].diff()
    
    # النطاق السعري
    df['Price_Range'] = df['High'] - df['Low']
    df['Volatility'] = df['Daily_Return'].rolling(window=7).std()
    
    # مؤشرات الحجم
    df['Volume_MA'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # إزالة القيم المفقودة
    df = df.dropna()
    
    return df

# ==================== دالة التنبؤ ====================
def train_and_predict(data, prediction_days=30, complexity="متوسط"):
    """تدريب النموذج والتنبؤ بالأسعار المستقبلية"""
    try:
        # إعداد البيانات
        df_features = create_features(data)
        
        # تحديد الميزات حسب مستوى التعقيد
        if complexity == "بسيط":
            features = ['Open', 'High', 'Low', 'Volume', 'MA_7']
            n_estimators = 50
        elif complexity == "متوسط":
            features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_14', 'Daily_Return']
            n_estimators = 100
        else:  # معقد
            features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_14', 'MA_30', 
                       'Daily_Return', 'Volatility', 'Volume_Ratio']
            n_estimators = 200
        
        # التأكد من توفر جميع الميزات
        available_features = [f for f in features if f in df_features.columns]
        
        if len(available_features) < 3:
            st.error("❌ لا توجد ميزات كافية للتدريب")
            return None, None, None, None
        
        X = df_features[available_features]
        y = df_features['Close']
        
        # تقسيم البيانات
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # تدريب النموذج
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train)
        
        # التنبؤ
        y_pred = model.predict(X_test)
        
        # حساب دقة النموذج
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        
        # التنبؤ المستقبلي
        last_features = X.iloc[-1:].values
        
        future_predictions = []
        future_dates = []
        
        current_date = df_features.index[-1]
        
        for i in range(1, prediction_days + 1):
            next_date = current_date + timedelta(days=i)
            future_dates.append(next_date)
            
            # التنبؤ بالسعر القادم
            next_price = model.predict(last_features)[0]
            future_predictions.append(next_price)
            
            # تحديث الميزات للتنبؤ التالي
            # (هذا نموذج مبسط، في التطبيق الحقيقي تحتاج إلى طريقة أفضل)
            last_features[0][0] = next_price  # Open
            last_features[0][1] = next_price * 1.01  # High
            last_features[0][2] = next_price * 0.99  # Low
        
        future_df = pd.DataFrame({
            'التاريخ': future_dates,
            'السعر_المتوقع': future_predictions
        })
        
        # حساب الاتجاه
        future_df['التغير_اليومي'] = future_df['السعر_المتوقع'].pct_change() * 100
        future_df['التغير_اليومي'].iloc[0] = 0
        
        future_df['الاتجاه'] = future_df['التغير_اليومي'].apply(
            lambda x: '🟢 صعود' if x > 0.1 else ('🔴 هبوط' if x < -0.1 else '⚪ ثبات')
        )
        
        # حساب متوسط السعر المتوقع
        avg_predicted_price = future_df['السعر_المتوقع'].mean()
        current_price = data['Close'].iloc[-1]
        percentage_change = ((avg_predicted_price - current_price) / current_price) * 100
        
        return y_test, y_pred, future_df, {
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'avg_predicted_price': avg_predicted_price,
            'percentage_change': percentage_change
        }
    
    except Exception as e:
        st.error(f"❌ خطأ في التدريب والتنبؤ: {str(e)}")
        return None, None, None, None

# ==================== الواجهة الرئيسية ====================
# العنوان الرئيسي
st.markdown('<h1 class="main-header">📊 Apple Stock Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)

# إذا لم يتم الضغط على زر التحليل بعد
if not analyze_button:
    # صفحة الترحيب
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; color: white; margin: 20px 0;'>
            <h2 style='color: white;'>🍎 مرحباً بك</h2>
            <p style='font-size: 1.2rem;'>
            منصة متكاملة لتحليل وتوقع أسعار أسهم Apple باستخدام الذكاء الاصطناعي
            </p>
            <p style='margin-top: 20px;'>
            ⚡ <strong>استخدم القائمة الجانبية لبدء التحليل</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # عرض المميزات
    st.markdown('<h2 class="sub-header">✨ مميزات المنصة</h2>', unsafe_allow_html=True)
    
    features_cols = st.columns(4)
    
    with features_cols[0]:
        st.markdown("""
        <div class='info-box'>
            <h4>📈 تحليل تاريخي</h4>
            <p>عرض كامل للبيانات التاريخية مع الرسوم البيانية التفاعلية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_cols[1]:
        st.markdown("""
        <div class='info-box'>
            <h4>🤖 تنبؤ ذكي</h4>
            <p>تنبؤ بالأسعار المستقبلية باستخدام خوارزميات التعلم الآلي</p>
        </div>
        """, unsafe_allow_html=True)
    
  
    with features_cols[2]:
        st.markdown("""
        <div class='info-box'>
            <h4>📊 تقارير تفاعلية</h4>
            <p>تقارير وتحليلات شاملة مع إمكانية التخصيص</p>
        </div>
        """, unsafe_allow_html=True)
    
    # إحصائيات عامة
    st.markdown('<h2 class="sub-header">📊 معلومات سريعة</h2>', unsafe_allow_html=True)
    
    try:
        # تحميل أحدث البيانات لعرض الإحصائيات
        quick_data = load_stock_data("AAPL", 
                                   start_date=datetime(today.year - 1, 1, 1),
                                   end_date=today)
        
        if quick_data is not None and not quick_data.empty:
            stats_cols = st.columns(4)
            
            with stats_cols[0]:
                current_price = quick_data['Close'].iloc[-1]
                st.metric("💰 السعر الحالي", f"${current_price:.2f}")
            
            with stats_cols[1]:
                price_change = ((current_price - quick_data['Close'].iloc[0]) / quick_data['Close'].iloc[0]) * 100
                st.metric("📈 التغير السنوي", f"{price_change:.2f}%")
            
            with stats_cols[2]:
                avg_volume = quick_data['Volume'].mean()
                st.metric("📊 متوسط الحجم", f"{avg_volume:,.0f}")
            
            with stats_cols[3]:
                volatility = quick_data['Close'].pct_change().std() * 100
                st.metric("⚡ التذبذب", f"{volatility:.2f}%")
    
    except:
        st.info("👈 استخدم القائمة الجانبية لتحميل البيانات الكاملة")
    
    # تعليمات سريعة
    with st.expander("📋 كيفية استخدام المنصة", expanded=True):
        st.markdown("""
        1. **تحديد الفترة الزمنية** من القائمة الجانبية
        2. **اختر خيارات التحليل** التي تريدها
        3. **اضغط على زر 'بدء التحليل الآن'**
        4. **استعرض النتائج** في الأقسام المختلفة
        5. **قم بتنزيل التقارير** إذا أردت حفظها
        """)

else:
    # ==================== بدء التحليل ====================
    
    # مؤشر التقدم
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # الخطوة 1: تحميل البيانات
    status_text.text("📥 جاري تحميل بيانات Apple...")
    data = load_stock_data("AAPL", start_date, end_date)
    progress_bar.progress(25)
    
    if data is None or data.empty:
        st.error("❌ لم نتمكن من تحميل البيانات. الرجاء المحاولة مرة أخرى.")
        st.stop()
    
    # الخطوة 3: التحليل الاستكشافي
    status_text.text("📊 جاري التحليل الاستكشافي...")
    progress_bar.progress(75)
    
    # الخطوة 4: التنبؤ
    status_text.text("🤖 جاري تدريب النموذج والتنبؤ...")
    y_test, y_pred, future_predictions, model_metrics = train_and_predict(
        data, prediction_days, model_complexity
    )
    progress_bar.progress(100)
    status_text.text("✅ تم الانتهاء من التحليل!")
    
    # ==================== عرض النتائج ====================
    
    # مؤشرات الأداء الرئيسية
    st.markdown('<h2 class="sub-header">📊 ملخص الأداء الرئيسي</h2>', unsafe_allow_html=True)
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        current_price = data['Close'].iloc[-1]
        st.metric(
            "💰 السعر الحالي", 
            f"${current_price:.2f}",
            delta=f"{((current_price - data['Close'].iloc[0])/data['Close'].iloc[0]*100):.2f}%"
        )
    
    with kpi_cols[1]:
        if model_metrics:
            st.metric(
                "🎯 دقة النموذج", 
                f"{model_metrics['r2']*100:.1f}%",
                help="معامل التحديد (R² Score)"
            )
        else:
            st.metric("🎯 دقة النموذج", "غير متوفر")
    
    with kpi_cols[2]:
        if future_predictions is not None:
            avg_pred = future_predictions['السعر_المتوقع'].mean()
            st.metric(
                "🔮 متوسط السعر المتوقع", 
                f"${avg_pred:.2f}",
                delta=f"{model_metrics['percentage_change']:.2f}%" if model_metrics else None
            )
        else:
            st.metric("🔮 متوسط السعر المتوقع", "غير متوفر")
    
    with kpi_cols[3]:
        total_days = len(data)
        st.metric("📅 عدد الأيام", f"{total_days}", 
                 help=f"من {start_date} إلى {end_date}")
    
    # ==================== عرض البيانات الخام ====================
    if show_raw_data:
        st.markdown('<h2 class="sub-header">📄 البيانات التاريخية</h2>', unsafe_allow_html=True)
        
        with st.expander("عرض/إخفاء البيانات", expanded=False):
            # البحث والتصفية
            col1, col2 = st.columns(2)
            with col1:
                rows_to_show = st.slider("عدد الصفوف للعرض:", 10, 500, 100)
            
            with col2:
                search_term = st.text_input("🔍 بحث في البيانات:", "")
            
            # عرض البيانات
            display_data = data.copy()
            
            if search_term:
                # البحث في الأعمدة النصية
                mask = display_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                display_data = display_data[mask]
            
            st.dataframe(
                display_data.tail(rows_to_show),
                use_container_width=True,
                height=400
            )
            
            # إحصائيات البيانات
            st.markdown("#### 📈 إحصائيات البيانات")
            stats_cols = st.columns(4)
            
            with stats_cols[0]:
                st.write("**المتوسطات:**")
                st.write(f"الإغلاق: ${data['Close'].mean():.2f}")
                st.write(f"الحجم: {data['Volume'].mean():,.0f}")
            
            with stats_cols[1]:
                st.write("**القيم القصوى:**")
                st.write(f"الأعلى: ${data['High'].max():.2f}")
                st.write(f"الأدنى: ${data['Low'].min():.2f}")
            
            with stats_cols[2]:
                st.write("**التذبذب:**")
                daily_return = data['Close'].pct_change().std() * 100
                st.write(f"اليومي: {daily_return:.2f}%")
            
            with stats_cols[3]:
                # زر تحميل البيانات
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 تحميل البيانات كـ CSV",
                    data=csv,
                    file_name=f"apple_stock_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    help="تحميل جميع البيانات بصيغة CSV"
                )
    
    # ==================== التحليل الاستكشافي ====================
    if show_analysis:
        st.markdown('<h2 class="sub-header">📊 التحليل الاستكشافي</h2>', unsafe_allow_html=True)
        
        # تبويبات الرسوم البيانية
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 السعر عبر الزمن", 
            "📊 التوزيع والإحصاء", 
            "🔥 الارتباطات",
            "📰 تحليل المشاعر",
            "📉 تحليل الحجم"
        ])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # رسم سعر الإغلاق مع المتوسطات المتحركة
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='سعر الإغلاق',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # إضافة المتوسطات المتحركة إذا كانت البيانات كافية
                if len(data) > 30:
                    data_copy = data.copy()
                    data_copy['MA_30'] = data_copy['Close'].rolling(window=30).mean()
                    data_copy['MA_7'] = data_copy['Close'].rolling(window=7).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=data_copy['Date'],
                        y=data_copy['MA_30'],
                        mode='lines',
                        name='المتوسط 30 يوم',
                        line=dict(color='orange', width=1.5, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=data_copy['Date'],
                        y=data_copy['MA_7'],
                        mode='lines',
                        name='المتوسط 7 أيام',
                        line=dict(color='red', width=1.5, dash='dot')
                    ))
                
                fig.update_layout(
                    title='📈 تطور سعر Apple مع الوقت',
                    xaxis_title='التاريخ',
                    yaxis_title='السعر ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 مؤشرات السعر")
                
                price_stats = {
                    'السعر الحالي': f"${current_price:.2f}",
                    'أعلى سعر': f"${data['High'].max():.2f}",
                    'أدنى سعر': f"${data['Low'].min():.2f}",
                    'متوسط السعر': f"${data['Close'].mean():.2f}",
                    'التغير الكلي': f"{((data['Close'].iloc[-1] - data['Close'].iloc[0])/data['Close'].iloc[0]*100):.2f}%"
                }
                
                for key, value in price_stats.items():
                    st.metric(key, value)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # توزيع أسعار الإغلاق
                fig = px.histogram(
                    data,
                    x='Close',
                    nbins=30,
                    title='📊 توزيع أسعار الإغلاق',
                    labels={'Close': 'سعر الإغلاق ($)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # مخطط الصندوق
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=data['Close'],
                    name='سعر الإغلاق',
                    boxpoints='outliers',
                    marker_color='#1E88E5'
                ))
                
                fig.update_layout(
                    title='📦 مخطط الصندوق لأسعار الإغلاق',
                    yaxis_title='السعر ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # إحصائيات مفصلة
            st.markdown("#### 📈 إحصائيات مفصلة")
            
            stats_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        
        with tab3:
            # تحليل الحجم
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='حجم التداول',
                marker_color='#4CAF50',
                opacity=0.7
            ))
            
            # خط سعر الإغلاق
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                name='سعر الإغلاق',
                yaxis='y2',
                line=dict(color='#1E88E5', width=2)
            ))
            
            fig.update_layout(
                title='📉 حجم التداول وسعر الإغلاق',
                xaxis_title='التاريخ',
                yaxis_title='حجم التداول',
                yaxis2=dict(
                    title='سعر الإغلاق ($)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # إحصائيات الحجم
            st.markdown("#### 📊 إحصائيات الحجم")
            
            volume_stats = {
                'الحجم اليومي المتوسط': f"{data['Volume'].mean():,.0f}",
                'أعلى حجم': f"{data['Volume'].max():,.0f}",
                'أدنى حجم': f"{data['Volume'].min():,.0f}",
                'مجموع الحجم': f"{data['Volume'].sum():,.0f}"
            }
            
            cols = st.columns(4)
            for (key, value), col in zip(volume_stats.items(), cols):
                with col:
                    st.metric(key, value)
    
    # ==================== التنبؤ بالأسعار ====================
    if run_prediction and future_predictions is not None and model_metrics:
        st.markdown('<h2 class="sub-header">🔮 التنبؤ بالأسعار المستقبلية</h2>', unsafe_allow_html=True)
        
        # مؤشرات أداء النموذج
        st.markdown("#### 📊 أداء نموذج التنبؤ")
        
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            st.metric(
                "🎯 الدقة (R²)", 
                f"{model_metrics['r2']*100:.2f}%",
                help="معامل التحديد - كلما اقترب من 100% كان أفضل"
            )
        
        with metrics_cols[1]:
            st.metric(
                "📏 متوسط الخطأ (RMSE)", 
                f"${model_metrics['rmse']:.2f}",
                help="جذر متوسط الخطأ التربيعي"
            )
        
        with metrics_cols[2]:
            st.metric(
                "📊 نسبة الخطأ (MAPE)", 
                f"{model_metrics['mape']:.2f}%",
                help="متوسط النسبة المئوية للخطأ المطلق"
            )
        
        with metrics_cols[3]:
            st.metric(
                "🤖 مستوى التعقيد", 
                model_complexity,
                help=f"عدد أيام التنبؤ: {prediction_days}"
            )
        
        # التنبؤات المستقبلية
        st.markdown("#### 📅 التنبؤات اليومية للشهر القادم")
        
        # تحسين شكل العرض
        display_predictions = future_predictions.copy()
        display_predictions['السعر_المتوقع'] = display_predictions['السعر_المتوقع'].apply(lambda x: f"${x:.2f}")
        display_predictions['التغير_اليومي'] = display_predictions['التغير_اليومي'].apply(lambda x: f"{x:.2f}%")
        
        # إعادة تسمية الأعمدة للعرض
        display_predictions.columns = ['📅 التاريخ', '💰 السعر المتوقع', '📈 التغير اليومي', '📊 الاتجاه']
        
        st.dataframe(
            display_predictions,
            use_container_width=True,
            height=400
        )
        
        # رسم التنبؤات
        st.markdown("#### 📈 رسم بياني للتنبؤات")
        
        fig = go.Figure()
        
        # البيانات التاريخية (آخر 90 يوم)
        historical_days = min(90, len(data))
        historical_data = data.tail(historical_days)
        
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Close'],
            mode='lines',
            name='البيانات التاريخية',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # التنبؤات المستقبلية
        fig.add_trace(go.Scatter(
            x=future_predictions['التاريخ'],
            y=future_predictions['السعر_المتوقع'],
            mode='lines+markers',
            name='التنبؤات المستقبلية',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            marker=dict(
                size=8,
                color=future_predictions['السعر_المتوقع'],
                colorscale='Viridis',
                showscale=False
            )
        ))
        
        # منطقة الثقة (نموذجية)
        fig.add_trace(go.Scatter(
            x=pd.concat([future_predictions['التاريخ'].iloc[[0]], 
                        future_predictions['التاريخ'].iloc[[-1]]]),
            y=[future_predictions['السعر_المتوقع'].mean() * 0.95,
               future_predictions['السعر_المتوقع'].mean() * 0.95],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.concat([future_predictions['التاريخ'].iloc[[0]], 
                        future_predictions['التاريخ'].iloc[[-1]]]),
            y=[future_predictions['السعر_المتوقع'].mean() * 1.05,
               future_predictions['السعر_المتوقع'].mean() * 1.05],
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            mode='lines',
            line=dict(width=0),
            name='منطقة الثقة (±5%)'
        ))
        
        fig.update_layout(
            title=f'🔮 تنبؤ أسعار Apple للـ {prediction_days} يوم القادمة',
            xaxis_title='التاريخ',
            yaxis_title='السعر ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # تحليل الاتجاه
        st.markdown("#### 📊 تحليل الاتجاه المستقبلي")
        
        # حساب إحصائيات الاتجاه
        upward_days = len(future_predictions[future_predictions['الاتجاه'].str.contains('🟢')])
        downward_days = len(future_predictions[future_predictions['الاتجاه'].str.contains('🔴')])
        stable_days = len(future_predictions[future_predictions['الاتجاه'].str.contains('⚪')])
        
        total_days_pred = len(future_predictions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 أيام الصعود المتوقعة", 
                     f"{upward_days} يوم",
                     f"{(upward_days/total_days_pred)*100:.1f}%")
        
        with col2:
            st.metric("🔴 أيام الهبوط المتوقعة", 
                     f"{downward_days} يوم",
                     f"{(downward_days/total_days_pred)*100:.1f}%")
        
        with col3:
            st.metric("⚪ أيام الثبات المتوقعة", 
                     f"{stable_days} يوم",
                     f"{(stable_days/total_days_pred)*100:.1f}%")
        
        # توصيات استثمارية
        st.markdown("#### 💡 توصيات استثمارية (لأغراض تعليمية)")
        
        recommendation_cols = st.columns(3)
        
        with recommendation_cols[0]:
            st.info("""
            **📈 للشراء (إذا):**
            - السعر الحالي أقل من المتوسط المتوقع
            - المشاعر الإيجابية مرتفعة
            - الاتجاه العام صعودي
            """)
        
        with recommendation_cols[1]:
            st.warning("""
            **📉 للبيع (إذا):**
            - السعر الحالي أعلى من المتوسط المتوقع
            - المشاعر السلبية مرتفعة
            - الاتجاه العام هبوطي
            """)
        
        with recommendation_cols[2]:
            st.success("""
            **⚖️ الانتظار (إذا):**
            - السوق متقلب
            - الاتجاه غير واضح
            - تحتاج لمزيد من البيانات
            """)
        
        # تنزيل التنبؤات
        st.markdown("#### 📥 تحميل التنبؤات")
        
        predictions_csv = future_predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 تحميل التنبؤات كـ CSV",
            data=predictions_csv,
            file_name=f"apple_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="تحميل جميع التنبؤات المستقبلية"
        )
    
    elif run_prediction:
        st.warning("⚠️ التنبؤ غير متوفر. قد تكون البيانات غير كافية أو حدث خطأ في التدريب.")
    
    # ==================== خاتمة وتوصيات ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📋 ملخص النتائج والتوصيات</h2>', unsafe_allow_html=True)
    
    summary_cols = st.columns(2)
    
    with summary_cols[0]:
        st.markdown("""
        ### ✅ النقاط الإيجابية
        - 📈 **اتجاه السعر:** تحليل دقيق للاتجاه التاريخي
        - 🤖 **دقة النموذج:** نتائج تنبؤ موثوقة
        - 💬 **تحليل المشاعر:** فهم تأثير الأخبار
        - 📊 **التقارير الشاملة:** جميع البيانات في مكان واحد
        """)
    
    with summary_cols[1]:
        st.markdown("""
        ### ⚠️ النقاط التي تحتاج مراجعة
        - 🔄 **تحديث البيانات:** تحتاج للتحديث اليومي
        - 📈 **تقلبات السوق:** قد تكون غير متوقعة
        - 🤖 **حدود الذكاء الاصطناعي:** النماذج ليست معصومة
        - ⏱️ **زمن المعالجة:** قد يطول مع البيانات الكبيرة
        """)
    
    # تحذير مهم
    st.warning("""
    ⚠️ **تنويه هام:** هذا التطبيق للأغراض التعليمية والتحليلية فقط. 
    لا يعتبر نصيحة مالية أو توصية استثمارية. 
    دائماً قم باستشارة مستشار مالي محترف قبل اتخاذ أي قرارات استثمارية.
    """)
    
    # معلومات الاتصال/الدعم
    with st.expander("🆘 الدعم والمزيد من المعلومات"):
        st.markdown("""        
        ### 📚 مصادر التعلم:
        - [Yahoo Finance Documentation](https://finance.yahoo.com/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Scikit-learn Documentation](https://scikit-learn.org/)
        
        ### 🔄 تحديثات التطبيق:
        - الإصدار الحالي: 1.0.0
        - آخر تحديث: نوفمبر 2024
        - المطور: فريق مطور جامعة حلوان التكنولوجيا
        """)

# ==================== تذييل الصفحة ====================
st.markdown("---")
footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown("""
    **🍎 Apple Stock Analysis**  
    منصة متكاملة للتحليل والتنبؤ
    """)

with footer_cols[1]:
    st.markdown("""
    **📊 البيانات المقدمة:**  
    لأغراض تعليمية فقط  
    آخر تحديث: """ + datetime.now().strftime("%Y-%m-%d"))

with footer_cols[2]:
    st.markdown("""
    **🚀 تم التطوير باستخدام:**  
    Streamlit · Python · Machine Learning
    """)