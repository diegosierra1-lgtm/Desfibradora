# 📋 Descripción de Códigos - Desfibradora

## 🎯 Scripts Principales

### 1. **analisis_prediccion_fallas.py**
Entrena un modelo Random Forest para detectar fallas en la desfibradora.
- Carga y limpia datos crudos
- Genera 8 características temporales (sin/cos de hora, día, mes)
- Divide datos 80/20 (entrenamiento/prueba)
- Entrena modelo con 100 árboles
- Guarda: `random_forest_model.joblib`, `scaler.joblib`, `Desfibradora_con_predicciones.csv`
- Genera gráficos de análisis

### 2. **predictor.py**
Interfaz para hacer predicciones con el modelo entrenado.
- Carga modelo y escalador
- Predice desde valores individuales (7 sensores)
- Genera automáticamente características temporales
- Modo demo, parámetros CLI, o lectura desde CSV
- Salida: Estado (Normal/Falla) + probabilidades + confianza

### 3. **monitor_realtime.py**
Monitor en tiempo real de la desfibradora.
- Simula lecturas de sensores
- Dashboard actualizado cada 5 segundos
- Historial de últimas 5 lecturas
- Alertas en tiempo real
- Reporte final con estadísticas

### 4. **analisis_tendencias.py**
Analiza tendencias de degradación de la máquina.
- Detecta patrones de cambio en sensores
- Visualiza gráficos de evolución
- Identifica deterioro vs normalidad

---

## 🔬 Scripts de Autoencoder (Detección de Anomalías)

### 5. **autoencoder_detector_fallas.py**
Autoencoder con TensorFlow/Keras para detección de anomalías.
- Red neuronal: 7 → 5 → 3 → 5 → 7
- Entrena solo con datos normales
- Detecta anomalías por error de reconstrucción
- Métricas: Precisión, Recall, F1-Score, ROC AUC
- Guarda: `autoencoder_modelo.h5`

### 6. **autoencoder_detector_fallas_sklearn.py**
Autoencoder con Scikit-Learn (MLPRegressor).
- Alternativa más ligera a TensorFlow
- Misma arquitectura: 7 → 5 → 3 → 5 → 7
- Sin supervisión (aprende patrones normales)
- Genera reporte completo con análisis comparativo
- Guarda: `autoencoder_sklearn_model.joblib`

### 7. **autoencoder_umbral_optimizado_sklearn.py**
Optimiza el umbral de detección del autoencoder.
- Compara 3 estrategias: Media+3σ, Percentil 75 (RECOMENDADO), Percentil 90
- Calcula matrices de confusión y métricas
- Recomienda mejor umbral
- Guarda: `autoencoder_optimized_model.joblib`

---

## 📚 Otros Archivos

### 8. **EJEMPLOS.py**
Ejemplos de uso de los scripts principales.
- Cómo predecir valores individuales
- Cómo procesar CSV
- Cómo interpretar resultados

---

## 📊 Archivos Generados

| Archivo | Origen | Descripción |
|---------|--------|-------------|
| `random_forest_model.joblib` | analisis_prediccion_fallas.py | Modelo Random Forest entrenado |
| `scaler.joblib` | analisis_prediccion_fallas.py | Normalizador (15 características) |
| `Desfibradora_con_predicciones.csv` | analisis_prediccion_fallas.py | Dataset con predicciones |
| `Importancia_caracteristicas.csv` | analisis_prediccion_fallas.py | Importancia de cada sensor |
| `Analisis_prediccion_fallas.png` | analisis_prediccion_fallas.py | Gráficos de desempeño |
| `autoencoder_modelo.h5` | autoencoder_detector_fallas.py | Modelo Keras guardado |
| `autoencoder_optimized_model.joblib` | autoencoder_umbral_optimizado_sklearn.py | Modelo Sklearn optimizado |
| `scaler_autoencoder.joblib` | autoencoder_umbral_optimizado_sklearn.py | Escalador para autoencoder |

---

## 🔄 Flujo de Trabajo Recomendado

```
1. analisis_prediccion_fallas.py
   ↓
   Genera: modelo, scaler, dataset predicciones
   ↓
2. predictor.py (usar modelo entrenado)
   ↓
   Predicciones en tiempo real
   ↓
3. monitor_realtime.py (monitoreo continuo)
   ↓
   Dashboard + alertas

Opcional:
→ autoencoder_detector_fallas_sklearn.py (detección no supervisada)
→ autoencoder_umbral_optimizado_sklearn.py (optimizar umbral)
```

---

## ⚡ Uso Rápido

```bash
# Entrenar modelo
python analisis_prediccion_fallas.py

# Hacer predicción
python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 --env_b 1.6 --accel_a 0.54 --vel_a 1.7 --env_a 0.99

# Procesar CSV
python predictor.py --csv datos_nuevos.csv

# Monitoreo en tiempo real
python monitor_realtime.py

# Detección de anomalías
python autoencoder_detector_fallas_sklearn.py
```

---

## 📋 Características Disponibles

**Sensores Físicos (7):**
- RPM rotor
- Aceleración CHUMACERA B
- Velocidad CHUMACERA B
- Envelope CHUMACERA B
- Aceleración CHUMACERA A
- Velocidad CHUMACERA A
- Envelope CHUMACERA A

**Características Temporales (8):**
- hora_sin, hora_cos (ciclo 24h)
- minuto_sin, minuto_cos (ciclo 60 min)
- dia_semana_sin, dia_semana_cos (ciclo semanal)
- dia_mes_normalizado (0-1)
- mes_normalizado (0-1)

**Total: 15 características para predicción**
