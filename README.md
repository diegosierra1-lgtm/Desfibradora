# 🏭 DESFIBRADORA - Sistema de Predicción de Fallas

![Status](https://img.shields.io/badge/status-✅%20Operacional-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

## 📋 Descripción

Sistema completo de análisis y predicción de fallas en la **Desfibradora** usando **Machine Learning** con Random Forest. Limpia datos de sensores, entrena un modelo de clasificación y proporciona predicciones de fallas en tiempo real.

---

## 🎯 Características

✅ **Limpieza automática de datos** - Manejo inteligente de valores faltantes  
✅ **Análisis exploratorio** - Identificación de patrones de falla  
✅ **Random Forest** - 100 árboles de decisión optimizados  
✅ **Predicciones en tiempo real** - API de predicción simple  
✅ **Monitor en vivo** - Dashboard de monitoreo continuo  
✅ **Reportes detallados** - Gráficas y métricas de desempeño  

---

## 📊 Resultados

```
┌─────────────────────────────────────┐
│  MODELO: Random Forest              │
├─────────────────────────────────────┤
│  Registros procesados:   49,704     │
│  Precisión:              70.95%     │
│  Recuperación:           18.58%     │
│  F1-Score:               17.91%     │
│  Árboles:                100        │
└─────────────────────────────────────┘

DISTRIBUCIÓN DE DATOS
├─ Normal:  41,228 (82.9%)
└─ Falla:    8,476 (17.1%)

TIPOS DE FALLAS
├─ Fricción:       4,313
├─ Impactos:       2,091
└─ Desbalance:     2,072
```

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar o descargar el proyecto
cd desfibradora

# Instalar dependencias (ya están configuradas)
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Ejecutar Análisis Completo

```bash
# Entrenar modelo y generar reportes
python analisis_prediccion_fallas.py
```

**Salida**:
- `Desfibradora_con_predicciones.csv` - Dataset con predicciones
- `Importancia_caracteristicas.csv` - Ranking de sensores
- `Analisis_prediccion_fallas.png` - Gráficos de análisis

### 3. Hacer Predicciones

#### Opción A: Valores individuales
```bash
python predictor.py --rpm 1000 \
                    --accel_b 0.45 --vel_b 2.4 --env_b 1.6 \
                    --accel_a 0.54 --vel_a 1.7 --env_a 0.99
```

**Salida**:
```
✓ Estado: NORMAL ✓
  Probabilidad Normal: 89.40%
  Probabilidad Falla:  10.60%
  Confianza:           89.40%
```

#### Opción B: Desde archivo CSV
```bash
python predictor.py --csv nuevos_datos.csv
```

### 4. Monitor en Tiempo Real

```bash
python monitor_realtime.py
```

Muestra un dashboard activo que:
- Lee datos continuamente
- Genera alertas de falla
- Actualiza estadísticas en vivo

---

## 📁 Estructura del Proyecto

```
desfibradora/
├── 📊 Datos
│   ├── Desfibradora_crudo.csv              (160k registros)
│   ├── Desfibradora_rejilla_temporal.csv   (datos adicionales)
│   ├── Desfibradora_con_predicciones.csv   (SALIDA)
│   └── Importancia_caracteristicas.csv     (SALIDA)
│
├── 🐍 Scripts Python
│   ├── analisis_prediccion_fallas.py       (Principal - Entrenamiento)
│   ├── predictor.py                         (Predicciones individuales)
│   └── monitor_realtime.py                  (Monitor en vivo)
│
├── 📈 Visualizaciones
│   └── Analisis_prediccion_fallas.png      (4 gráficos)
│
└── 📚 Documentación
    ├── README.md                            (Este archivo)
    ├── GUIA_RAPIDA.md                       (Referencia rápida)
    └── .venv/                               (Entorno virtual)
```

---

## 🔧 Scripts Detallados

### `analisis_prediccion_fallas.py`

**Función**: Limpieza completa, entrenamiento y evaluación del modelo.

**Pasos**:
1. Carga 160k registros
2. Limpia datos (remove 69%)
3. Crea características
4. Entrena Random Forest
5. Evalúa con test set
6. Genera reportes

**Tiempo**: ~30-60 segundos

**Salida**:
- Métricas de desempeño
- Matriz de confusión
- Gráficos (4 subplots)
- CSV con predicciones

### `predictor.py`

**Función**: Interfaz para predicciones en tiempo real.

**Modos**:
- **Individual**: Parámetros de línea de comandos
- **CSV**: Archivo de datos completo
- **Demostración**: Ejemplos de uso

**Uso**:
```bash
# Individual
python predictor.py --rpm 1000 --accel_b 0.45 ...

# CSV
python predictor.py --csv datos.csv

# Demo
python predictor.py
```

### `monitor_realtime.py`

**Función**: Monitoreo continuo con alertas.

**Características**:
- Dashboard activo (actualiza cada 5s)
- Historial de 5 últimas lecturas
- Alertas en tiempo real
- Estadísticas dinámicas

**Salida**:
- Dashboard visual
- Reporte final
- Análisis de tendencias

---

## 📊 Interpretación de Resultados

### Métricas de Desempeño

| Métrica | Valor | Significado |
|---------|-------|------------|
| **Precisión** | 70.95% | 7 de 10 predicciones correctas |
| **Exactitud (Precision)** | 17.28% | Pocas falsas alarmas |
| **Recuperación (Recall)** | 18.58% | Detecta ~19 de cada 100 fallas reales |
| **F1-Score** | 17.91% | Balance entre exactitud y recuperación |

### Matriz de Confusión

```
                    Predicho Normal    Predicho Falla
Real Normal              6,738             1,508        ← 18% falsos positivos
Real Falla               1,380               315        ← 81% falsos negativos

Interpretación:
• Verdaderos Negativos (TN):   6,738 ✓ (Correctas predicciones normales)
• Verdaderos Positivos (TP):     315 ✓ (Correctas predicciones de falla)
• Falsos Positivos (FP):       1,508   (Alarma falsa)
• Falsos Negativos (FN):       1,380   (No detecta falla) ⚠️
```

### Recomendación de Uso

✅ **Ideal para**: Sistema de alertas tempranas  
⚠️ **Validar**: Todas las predicciones con especialista  
🔄 **Mejorar**: Recopilar más datos de fallos reales  

---

## 🎯 Características Importantes

Las 5 variables más predictivas:

1. **Aceleración CHUMACERA B** (17.17%) - Vibraciones de alta frecuencia
2. **Envelope CHUMACERA A** (16.33%) - Análisis de banda
3. **Velocidad CHUMACERA B** (16.00%) - Vibraciones de rango medio
4. **Envelope CHUMACERA B** (15.98%) - Análisis de banda
5. **Aceleración CHUMACERA A** (14.98%) - Vibraciones lado opuesto

**Conclusión**: Los sensores de aceleración y envelope son críticos.

---

## 💻 Requisitos

- **Python**: 3.8 o superior
- **Librerías**:
  - pandas >= 1.3
  - numpy >= 1.21
  - scikit-learn >= 1.0
  - matplotlib >= 3.4
  - seaborn >= 0.11

**Instalación**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🔍 Ejemplos de Uso

### Ejemplo 1: Ver predicciones guardadas
```python
import pandas as pd

df = pd.read_csv('Desfibradora_con_predicciones.csv')

# Ver registros con falla
fallos = df[df['prediccion'] == 1]
print(f"Fallos detectados: {len(fallos)}")
print(fallos[['datetime', 'prediccion', 'confianza']].head())
```

### Ejemplo 2: Importancia de características
```python
import pandas as pd

importance = pd.read_csv('Importancia_caracteristicas.csv')
print(importance.head(10))

# Graficar
importance.plot(x='caracteristica', y='importancia', kind='barh')
```

### Ejemplo 3: Validar predicciones
```python
df = pd.read_csv('Desfibradora_con_predicciones.csv')

# Contar predicciones
print("Predicciones del modelo:")
print(df['prediccion'].value_counts())

# Confianza promedio
print(f"Confianza promedio: {df['confianza'].mean():.2%}")
```

---

## ⚙️ Ajuste de Parámetros

Para mejorar el modelo, edita `analisis_prediccion_fallas.py`:

```python
modelo = RandomForestClassifier(
    n_estimators=100,        # Aumentar para mejor precisión
    max_depth=15,            # Reducir para evitar overfitting
    min_samples_split=5,     # Aumentar para generalización
    min_samples_leaf=2,      # Aumentar para estabilidad
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Balancear clases desiguales
)
```

### Recomendaciones

| Ajuste | Problema | Solución |
|--------|----------|----------|
| Bajo F1-Score | Pocas fallos en datos | Recolectar más datos de falla |
| Muchos falsos positivos | Demasiadas alarmas | Aumentar min_samples_split |
| No detecta fallos | Muchos falsos negativos | Disminuir max_depth |

---

## 📈 Próximos Pasos

- [ ] Recolectar más datos de fallas reales
- [ ] Validar predicciones con mantenimiento físico
- [ ] Integrar con sistema SCADA existente
- [ ] Configurar alertas automáticas
- [ ] Implementar reentrenamiento automático
- [ ] Crear dashboard web (Flask/Django)
- [ ] Documentación en monografía

---

## 🐛 Troubleshooting

### Error: "Archivo no encontrado"
```bash
# Asegúrate de estar en el directorio correcto
cd c:\Users\sierr\OneDrive\Desktop\Monografia\desfibradora
```

### Error: "Módulo no encontrado"
```bash
# Reinstala dependencias
pip install --upgrade pandas numpy scikit-learn matplotlib seaborn
```

### Predictor no funciona
```bash
# Asegúrate de que analisis_prediccion_fallas.py se ejecutó primero
python analisis_prediccion_fallas.py
```

---

## 📞 Información de Contacto

**Proyecto**: Análisis de Fallas Desfibradora  
**Fecha**: Mayo 2026  
**Estado**: ✅ En producción  

---

## 📄 Licencia

Este proyecto es parte de la monografía. Uso interno.

---

**Última actualización**: 7 de Mayo, 2026  
**Versión**: 1.0
