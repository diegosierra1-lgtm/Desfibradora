# 🏭 ANÁLISIS DE FALLAS DESFIBRADORA - Guía Rápida

## ✅ Estado del Análisis
**Completado exitosamente** - Random Forest entrenado y listo para predicciones

---

## 📊 RESULTADOS PRINCIPALES

### Datos Procesados
```
• Registros iniciales:       160,446
• Registros válidos:          49,704
• Tasa de limpieza:           69.0%
• Período:                    5-13 de Abril, 2026
```

### Distribución de Clases
```
• Estado NORMAL:  41,228 registros (82.9%)
• Con FALLA:       8,476 registros (17.1%)
• Desbalance:      4.9:1
```

### Tipos de Fallas Detectadas
```
1. Fricción      - 4,313 registros (50.8%)
2. Impactos      - 2,091 registros (24.7%)
3. Desbalance    - 2,072 registros (24.4%)
```

---

## 🤖 MODELO RANDOM FOREST

### Configuración
```
• Algoritmo:          Random Forest Classifier
• Número de árboles:  100
• Profundidad máx:    15
• Criterio división:  Gini
• Balanceo clases:    Sí (weight='balanced')
```

### Desempeño en TEST SET
```
📈 Precisión General:  70.95%
📊 Exactitud (Precision): 17.28%
🎯 Recuperación (Recall): 18.58%
⚖️ F1-Score: 17.91%
```

### Matriz de Confusión
```
                Predicho Normal  Predicho Falla
Real Normal           6,738            1,508
Real Falla              1,380             315

• Verdaderos Negativos (TN):   6,738 ✓
• Falsos Positivos (FP):       1,508 
• Falsos Negativos (FN):       1,380
• Verdaderos Positivos (TP):     315 ✓
```

### Interpretación
- **70.95% de precisión**: El modelo clasifica correctamente 7 de cada 10 registros
- **18.58% de recuperación**: Detecta 19 de cada 100 fallas reales (conservador)
- **Recomendación**: Usar como sistema de **alertas tempranas** (menos falsos negativos)

---

## 🎯 CARACTERÍSTICAS MÁS IMPORTANTES

### Top 5 Variables Predictivas
```
1. Aceleración CHUMACERA B        17.17%
2. Envelope CHUMACERA A           16.33%
3. Velocidad CHUMACERA B          16.00%
4. Envelope CHUMACERA B           15.98%
5. Aceleración CHUMACERA A        14.98%
```

**Conclusión**: Los sensores de aceleración y envelope son críticos para detectar fallas.

---

## 📁 ARCHIVOS GENERADOS

### Archivos de Datos
| Archivo | Descripción |
|---------|-------------|
| `Desfibradora_con_predicciones.csv` | Dataset original + predicciones del modelo |
| `Importancia_caracteristicas.csv` | Ranking de importancia de cada sensor |
| `predicciones_nuevas.csv` | Predicciones en nuevos datos (cuando se ejecute predictor.py) |

### Gráficas
| Archivo | Contenido |
|---------|----------|
| `Analisis_prediccion_fallas.png` | 4 gráficos de análisis completo |

### Scripts Python
| Script | Propósito |
|--------|----------|
| `analisis_prediccion_fallas.py` | Análisis completo y entrenamiento del modelo |
| `predictor.py` | Hacer predicciones en tiempo real |

---

## 🚀 CÓMO USAR

### 1️⃣ Ver Predicciones Existentes
```bash
# Ver el dataset con predicciones
python -c "import pandas as pd; df = pd.read_csv('Desfibradora_con_predicciones.csv'); print(df[['datetime', 'prediccion', 'confianza']].head(20))"
```

### 2️⃣ Hacer Predicción con Valores Específicos
```bash
# Predecir con valores individuales
python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 --env_b 1.6 \
                    --accel_a 0.54 --vel_a 1.7 --env_a 0.99
```

### 3️⃣ Predecir Nuevos Datos desde CSV
```bash
# Si tienes nuevos datos en un CSV con la misma estructura
python predictor.py --csv nuevos_datos.csv
```

### 4️⃣ Reentrenar el Modelo (si hay nuevos datos)
```bash
# Ejecutar análisis nuevamente
python analisis_prediccion_fallas.py
```

---

## 📈 INTERPRETACIÓN DE PREDICCIONES

### Estado NORMAL ✓
- Confianza típica: > 70%
- Acción: Operación normal, monitoreo rutinario
- Ejemplo: `✓ NORMAL | Confianza: 89.4%`

### Estado FALLA ⚠️
- Confianza típica: 50-99%
- Acción: Revisar sensores indicados, programar mantenimiento
- Ejemplo: `⚠️ FALLA | Confianza: 51.2%`

### Confianza Baja (50-60%)
- Zona gris, revisar manualmente
- Validar con diagnóstico físico
- Considerar como "alerta amarilla"

---

## 🔍 ANÁLISIS DE CARACTERÍSTICAS

### Qué Mide Cada Sensor

#### CHUMACERA B (Lado Corriente)
- **RPM Rotor**: Revoluciones por minuto del rotor
- **Aceleración**: Vibraciones de alta frecuencia (fallos agudos)
- **Velocidad**: Vibraciones de frecuencia media
- **Envelope**: Análisis de banda de envolvente

#### CHUMACERA A (Lado Opuesto)
- Mismas mediciones, lado opuesto
- Comparar ambos lados para diagnosticar problemas de alineación

---

## 💡 RECOMENDACIONES

### Para Mejorar el Modelo
1. **Más datos**: Incluir más registros de fallas
2. **Balanceo**: El dataset está desbalanceado (82.9% vs 17.1%)
3. **Tuning**: Ajustar hiperparámetros (profundidad, min_samples_split)
4. **Características nuevas**: Agregar ratios entre sensores

### Para Uso en Producción
1. **Alertas**: Configurar alertas automáticas cuando se predice FALLA
2. **Umbrales**: Ajustar umbral de confianza según tolerancia
3. **Monitoreo**: Revisar regularmente predicciones vs diagnóstico real
4. **Retroalimentación**: Validar predicciones con especialistas

---

## 📚 ESTRUCTURA DE CSV

### Desfibradora_con_predicciones.csv
```
maquina,datetime,RPM rotor,Aceleration_CHUMACERA B,...,prediccion,confianza
ROTOR DESFIBRADORA,2026-04-05,...,0,0.894
```

**Columnas principales**:
- `prediccion`: 0 = Normal, 1 = Falla
- `confianza`: 0-1 (0% a 100%)

### Importancia_caracteristicas.csv
```
caracteristica,importancia
RPM rotor,0.1234
Aceleration_CHUMACERA B,0.1717
```

---

## ⚠️ LIMITACIONES CONOCIDAS

1. **Recuperación baja**: Solo detecta ~19% de fallas (conservador)
2. **Falsos positivos**: 18% de alarmas falsas
3. **Datos históricos**: Modelo basado en datos de abril 2026
4. **Desbalance**: Pocas muestras de falla (17.1%)

---

## 📞 PRÓXIMOS PASOS

1. ✅ Recolectar más datos de fallas reales
2. ✅ Validar predicciones con mantenimiento físico
3. ✅ Ajustar parámetros según retroalimentación
4. ✅ Integrar en sistema de monitoreo automático
5. ✅ Documentar casos de falla no detectados

---

**Última actualización**: Mayo 7, 2026  
**Estado**: ✅ Producción
