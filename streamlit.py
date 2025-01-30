import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

st.title('Streamlit-Предсказатель значений по файлу с временным рядом и метрики ')
st.write('Загрузите ваш датасет по кнопке слева. Примечание: в датасете должна быть хотя бы одна колонка типа datetime ')
# st.dataframe(pd.DataFrame(columns=['Список', 'Параметров', 'Укажите', 'Какой', 'Параметр', 'Предсказать']))
uploader = st.sidebar.file_uploader('Загрузите csv-файл')

if uploader is not None:
    df = pd.read_csv(uploader)
    # Проверяем, есть ли хотя бы одна колонка, которая может быть преобразована в временные метки
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            date_columns.append(col)  # Если преобразование успешно, добавляем колонку в список
        except Exception:
            continue  # Если ошибка, пропускаем колонку
    
    if len(date_columns) > 0:
        st.write('Ваш файл:')
        st.dataframe(df)
        date = st.selectbox('Укажите название колонки с временными метками', options=date_columns, index=None)
        target = st.selectbox('Напишите название колонки, которую хотите предсказать', options=df.columns, index=None)
    else:
        st.error('В датасете нет колонки с временными метками.')

    lags = st.slider('Введите число lag\'ов', value=52)
    forecast_num = st.slider('Выбирите период на какой хотите предсказать', min_value=1)
    if date and target and lags and forecast_num is not None:
        date = pd.to_datetime(df[date])
        df1 = df.groupby(date)[target].sum()
        df1.sort_index()
        st.write('Подготовленный датасет для обработки:')   
        df1 = pd.DataFrame(df1)
        st.dataframe(data=df1, width=1000)


        st.write(f'График {target} по {date.name}')
        st.line_chart(df1)

        st.write('Обработка датасета...')
        lag = lags
        df_copy = df1.copy()
        for i in range(1, lag + 1):
            df_copy[f'lag_{i}'] = df_copy[target].shift(i)

        df_copy= df_copy.dropna()
        X, y  = df_copy.drop(target, axis=1), df_copy[target]


        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, shuffle=False)
        model_rf.fit(x_train, y_train)

        y_pred_rf = model_rf.predict(x_valid)
        mae_rf = mean_absolute_error(y_valid, y_pred_rf)
        mape_rf = mean_absolute_percentage_error(y_valid, y_pred_rf)
        print(f"MAE_rf: {mae_rf:.2f}")
        print(f"MAPE_rf: {mape_rf:.2f}")


        forecast = forecast_num  # период  предсказания
        future_dates = pd.date_range(start=X.index[-1], periods=forecast + 1, freq="W")[1:]

        future_df_rf = pd.DataFrame(index=future_dates, columns=X.columns)
        future_df_rf.iloc[0] = X.iloc[-1]

        # Пошаговое предсказание
        for i in range(1, forecast):
            X_future_rf = future_df_rf.iloc[i - 1][:].values.reshape(1, -1)  # Берем текущие лаги
            predicted_value_rf = model_rf.predict(X_future_rf)[0]  # Делаем предсказание

            future_df_rf.iloc[i, 0] = predicted_value_rf
            future_df_rf.iloc[i, 1:] = X_future_rf[0][:-1]

        predicted_forecast_rf = model_rf.predict(future_df_rf)
        forecast_rf = pd.DataFrame(data=predicted_forecast_rf, index=future_df_rf.index, columns=['Forecast'])
        st.write('Готовый датасет')
        csv = forecast_rf.to_csv(index=False).encode('utf-8')
        st.dataframe(data=forecast_rf)
        st.write('График предсказаний')
        if len(forecast_rf) > 2:
            st.line_chart(forecast_rf)
        st.download_button(
        label="Скачать датасет предсказания",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )