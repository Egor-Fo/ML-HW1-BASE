## 1. Что было сделано
- Были предобраотны данные об автомобилях для построения моделей
- Был обучен ряд моделей - Линейная регрессия, Lasso, Ridge, ElasticNet.  
- Сравнил модели по основным метрикам - MSE и R2, создана собственная бизнес-метрика
- По итогу исследования был сделан сервер для построения лучшей модели на основе бизнес-метрики  

## 2. Результаты
- Лучшая модель: ElasticNet (R² = 0.55, business_metric = 0.247).
- ПО R2 - RIDGE (0.62)
- Наибольший буст качества дала обработка категориальных признаков через OneHot-кодирование.  

Общие резулуьтаты:
Линейная регрессия
Тестовая выборка - business_metric: 0.22
Лассо
Тестовая выборка - business_metric: 0.239
Эластик
Тестовая выборка - business_metric: 0.247
Ridge
Тестовая выборка - business_metric: 0.224

## 4. Наибольший буст в качестве
- Наибольшее влияние оказывает Max_power
- Подбор параметров grid search, обработка категориальных признаков

## 3. Что не удалось
- Не удалось получить хорошую точность R2 > 0.8  
- Не удалось использовать "torque" 

Дополнительные файлы - 
API_HW1.postman_collection.json - запросы к сервису
