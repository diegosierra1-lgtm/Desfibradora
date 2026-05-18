"""
DESFIBRADORA - Ejemplos de Uso
==============================
Colección de ejemplos prácticos para usar el sistema
"""

# ============================================================================
# EJEMPLO 1: Ver predicciones guardadas
# ============================================================================

import pandas as pd
import numpy as np

print("EJEMPLO 1: Ver predicciones guardadas")
print("-" * 80)

# Cargar datos con predicciones
df = pd.read_csv('Desfibradora_con_predicciones.csv')

# Ver columnas disponibles
print("\nColumnas disponibles:")
print(df.columns.tolist())

# Ver primeras 5 predicciones
print("\nPrimeras 5 predicciones:")
print(df[['datetime', 'prediccion', 'confianza', 'tipo_falla']].head())

# Filtrar solo fallos
fallos = df[df['prediccion'] == 1]
print(f"\nTotal de fallos detectados: {len(fallos)}")
print("\nÚltimos 5 fallos:")
print(fallos[['datetime', 'tipo_falla', 'confianza']].tail())

# ============================================================================
# EJEMPLO 2: Análisis de importancia de características
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 2: Importancia de características")
print("-" * 80)

importance = pd.read_csv('Importancia_caracteristicas.csv')

print("\nTop 5 características:")
print(importance.head())

print("\nImportancia acumulada:")
importance['importancia_acumulada'] = importance['importancia'].cumsum()
print(importance)

# ============================================================================
# EJEMPLO 3: Estadísticas por tipo de falla
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 3: Estadísticas por tipo de falla")
print("-" * 80)

print("\nDistribución de tipos de falla:")
tipos = df['tipo_falla'].value_counts()
print(tipos)

print("\nPorcentaje:")
porcentaje = (df['tipo_falla'].value_counts() / len(df) * 100)
print(porcentaje.round(2))

# ============================================================================
# EJEMPLO 4: Analizar tendencias temporales
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 4: Tendencias temporales")
print("-" * 80)

# Convertir a datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Extraer hora
df['hora'] = df['datetime'].dt.hour

# Fallos por hora
fallos_por_hora = df[df['prediccion'] == 1].groupby('hora').size()
print("\nFallos detectados por hora:")
print(fallos_por_hora)

# Tasa de fallo por hora
tasa_fallo = (df[df['prediccion'] == 1].groupby('hora').size() / 
              df.groupby('hora').size() * 100)
print("\nTasa de fallo por hora (%):")
print(tasa_fallo.round(2))

# ============================================================================
# EJEMPLO 5: Análisis de confianza
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 5: Análisis de confianza")
print("-" * 80)

print("\nEstadísticas de confianza:")
print(df['confianza'].describe())

print("\nPredicciones con baja confianza (<60%):")
bajo_confianza = df[df['confianza'] < 0.6]
print(f"Total: {len(bajo_confianza)} ({len(bajo_confianza)/len(df)*100:.1f}%)")
print(bajo_confianza[['datetime', 'prediccion', 'confianza']].head(10))

# ============================================================================
# EJEMPLO 6: Correlación entre sensores
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 6: Correlación entre sensores")
print("-" * 80)

sensor_cols = [
    'RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
    'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 
    'Velocity_CHUMACERA A', 'Envelope_CHUMACERA A'
]

print("\nCorrelación entre sensores:")
correlacion = df[sensor_cols].corr()
print(correlacion)

# ============================================================================
# EJEMPLO 7: Validación simple del modelo
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 7: Validación del modelo")
print("-" * 80)

# Cargar datos
df = pd.read_csv('Desfibradora_con_predicciones.csv')

# Cantidad de predicciones
total = len(df)
normales = (df['prediccion'] == 0).sum()
fallos = (df['prediccion'] == 1).sum()

print(f"\nTotal de registros: {total}")
print(f"Predicciones normales: {normales} ({normales/total*100:.1f}%)")
print(f"Predicciones de falla: {fallos} ({fallos/total*100:.1f}%)")

print(f"\nConfianza promedio: {df['confianza'].mean():.2%}")
print(f"Confianza máxima: {df['confianza'].max():.2%}")
print(f"Confianza mínima: {df['confianza'].min():.2%}")

# ============================================================================
# EJEMPLO 8: Exportar predicciones a nuevos formatos
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 8: Exportar predicciones")
print("-" * 80)

# Solo fallos
df_fallos = df[df['prediccion'] == 1]
df_fallos.to_csv('fallos_detectados.csv', index=False)
print("✓ Archivo guardado: fallos_detectados.csv")

# Resumen diario
df['fecha'] = df['datetime'].dt.date
resumen_diario = df.groupby('fecha').agg({
    'prediccion': lambda x: (x == 1).sum(),
    'confianza': 'mean'
}).rename(columns={'prediccion': 'fallos_detectados', 'confianza': 'confianza_promedio'})

resumen_diario.to_csv('resumen_diario.csv')
print("✓ Archivo guardado: resumen_diario.csv")

# ============================================================================
# EJEMPLO 9: Predicción con nuevos datos
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 9: Predicción con nuevos valores")
print("-" * 80)

# Este ejemplo usa el script predictor.py en la terminal
print("""
Ejecutar en terminal:

# Predicción individual
python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 --env_b 1.6 \\
                    --accel_a 0.54 --vel_a 1.7 --env_a 0.99

# Predicción desde CSV
python predictor.py --csv nuevos_datos.csv
""")

# ============================================================================
# EJEMPLO 10: Monitor en tiempo real
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 10: Monitor en tiempo real")
print("-" * 80)

print("""
Ejecutar en terminal:

python monitor_realtime.py

Esto mostrará:
- Dashboard en vivo
- Últimas 5 lecturas
- Alertas activas
- Estadísticas dinámicas
""")

# ============================================================================
# EJEMPLO 11: Análisis personalizado
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 11: Análisis personalizado")
print("-" * 80)

df = pd.read_csv('Desfibradora_con_predicciones.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Fallos en rango de tiempo específico
inicio = '2026-04-10'
fin = '2026-04-12'
df_periodo = df[(df['datetime'].dt.date >= pd.to_datetime(inicio).date()) &
                (df['datetime'].dt.date <= pd.to_datetime(fin).date())]

print(f"\nFallos entre {inicio} y {fin}: {(df_periodo['prediccion'] == 1).sum()}")

# Estadísticas por CHUMACERA
for sensor in ['CHUMACERA B', 'CHUMACERA A']:
    print(f"\n{sensor}:")
    cols = [c for c in sensor_cols if sensor in c]
    print(f"  Promedio: {df[cols].mean().mean():.4f}")
    print(f"  Máximo: {df[cols].max().max():.4f}")
    print(f"  Mínimo: {df[cols].min().min():.4f}")

# ============================================================================
# EJEMPLO 12: Crear visualizaciones personalizadas
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 12: Visualizaciones personalizadas")
print("-" * 80)

print("""
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Desfibradora_con_predicciones.csv')

# Gráfico 1: Distribución de predicciones
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['prediccion'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Distribución de Predicciones')
axes[0].set_xticklabels(['Normal', 'Falla'], rotation=0)

# Gráfico 2: Confianza por predicción
df.boxplot(column='confianza', by='prediccion', ax=axes[1])
axes[1].set_title('Confianza por Predicción')

plt.tight_layout()
plt.savefig('visualizaciones_personalizadas.png', dpi=300)
plt.show()
""")

# ============================================================================
# RESUMEN DE EJEMPLOS
# ============================================================================

print("\n" + "="*80)
print("RESUMEN")
print("="*80)

print("""
✓ Ejemplo 1: Ver predicciones básicas
✓ Ejemplo 2: Analizar importancia de características
✓ Ejemplo 3: Estadísticas por tipo de falla
✓ Ejemplo 4: Tendencias temporales
✓ Ejemplo 5: Análisis de confianza
✓ Ejemplo 6: Correlación entre sensores
✓ Ejemplo 7: Validación del modelo
✓ Ejemplo 8: Exportar predicciones
✓ Ejemplo 9: Nuevas predicciones
✓ Ejemplo 10: Monitor en tiempo real
✓ Ejemplo 11: Análisis personalizado
✓ Ejemplo 12: Visualizaciones personalizadas

PRÓXIMOS PASOS:
1. Ejecutar analisis_prediccion_fallas.py
2. Explorar Desfibradora_con_predicciones.csv
3. Usar predictor.py para nuevas predicciones
4. Ejecutar monitor_realtime.py para monitoreo
5. Adaptar ejemplos a tus necesidades
""")

print("\n" + "="*80)
print("¡Listo para usar!")
print("="*80)
