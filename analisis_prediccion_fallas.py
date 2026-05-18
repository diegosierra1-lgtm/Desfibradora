"""
DESFIBRADORA - Análisis y Predicción de Fallas con Random Forest
=====================================================================
Script para limpiar, organizar y predecir fallas en la desfibradora
usando datos de sensores y Machine Learning.

Autor: Diego Sierra
Fecha: 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Añadir para persistencia

warnings.filterwarnings('ignore')

print("="*80)
print("DESFIBRADORA - Sistema de Predicción de Fallas")
print("="*80)

# ============================================================================
# 1. CARGAR Y LIMPIAR DATOS
# ============================================================================
print("\n[1] Cargando y limpiando datos...")

try:
    # Cargar datos
    df = pd.read_csv('Desfibradora_crudo.csv')
    print(f"✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Información inicial
    print(f"\nEstructura inicial:")
    print(f"  - Máquina: {df['maquina'].unique()}")
    print(f"  - Rango temporal: {df['datetime'].min()} a {df['datetime'].max()}")
    print(f"  - Valores faltantes: {df.isnull().sum().sum()} de {df.size}")
    
except Exception as e:
    print(f"✗ Error cargando datos: {e}")
    exit()

# ============================================================================
# 2. PREPROCESAMIENTO
# ============================================================================
print("\n[2] Preprocesamiento de datos...")

# Convertir datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# ============================================================================
# 2.1 CARACTERÍSTICAS TEMPORALES - NORMALIZACIÓN DE FECHAS
# ============================================================================
print("\n  [2.1] Extrayendo características temporales...")

# Extraer componentes de fecha
df['hora'] = df['datetime'].dt.hour
df['minuto'] = df['datetime'].dt.minute
df['dia_semana'] = df['datetime'].dt.dayofweek  # 0=lunes, 6=domingo
df['dia_mes'] = df['datetime'].dt.day
df['mes'] = df['datetime'].dt.month

# NORMALIZACIÓN CORRECTA de características temporales
# Usar sin/cos para capturar periodicidad (23:59 cercano a 00:01)
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

df['minuto_sin'] = np.sin(2 * np.pi * df['minuto'] / 60)
df['minuto_cos'] = np.cos(2 * np.pi * df['minuto'] / 60)

df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)

df['dia_mes_normalizado'] = (df['dia_mes'] - 1) / 30  # Escala 0-1
df['mes_normalizado'] = (df['mes'] - 1) / 11          # Escala 0-1

print("    ✓ Características temporales creadas:")
print("      - hora_sin, hora_cos (ciclo 24h)")
print("      - minuto_sin, minuto_cos (ciclo 60min)")
print("      - dia_semana_sin, dia_semana_cos (ciclo semanal)")
print("      - dia_mes_normalizado (0-1)")
print("      - mes_normalizado (0-1)")

# Crear copia para limpiar
df_clean = df.copy()

# Separar columnas de sensores y diagnóstico
sensor_cols = [
    'RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
    'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 'Velocity_CHUMACERA A',
    'Envelope_CHUMACERA A',
    # Nuevas características temporales
    'hora_sin', 'hora_cos', 'minuto_sin', 'minuto_cos',
    'dia_semana_sin', 'dia_semana_cos', 'dia_mes_normalizado', 'mes_normalizado'
]

falla_cols = [col for col in df.columns if col not in 
              ['maquina', 'datetime', 'hora', 'minuto', 'dia_semana', 'dia_mes', 'mes'] + sensor_cols]

print(f"✓ Columnas de sensores + temporales ({len(sensor_cols)}): {len(sensor_cols)}")
print(f"  - Sensores físicos: 7")
print(f"  - Características temporales: {len(sensor_cols) - 7}")
print(f"✓ Columnas de diagnóstico de fallas: {len(falla_cols)}")

# Estrategia de limpieza: llenar valores faltantes
print("\nEstrategia de limpieza:")
print("  - Valores faltantes en sensores: llenar con 0")
print("  - Valores faltantes en diagnóstico: llenar con 0")
print("  - Valores faltantes temporales: llenar con media")
print("  - Remover duplicados y valores inválidos")

# Llenar NaN apropiadamente
# Sensores y diagnóstico: llenar con 0
sensor_fisicos = ['RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
                  'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 'Velocity_CHUMACERA A',
                  'Envelope_CHUMACERA A']
df_clean[sensor_fisicos] = df_clean[sensor_fisicos].fillna(0)
df_clean[falla_cols] = df_clean[falla_cols].fillna(0)

# Características temporales: llenar con la media (no deben tener NaN si datetime es válido)
tempo_cols = [col for col in sensor_cols if col not in sensor_fisicos]
df_clean[tempo_cols] = df_clean[tempo_cols].fillna(df_clean[tempo_cols].mean())

# Remover duplicados completos
df_clean = df_clean.drop_duplicates()

# Remover filas donde todos los sensores sean 0
mask_valid = (df_clean[sensor_cols] != 0).any(axis=1)
df_clean = df_clean[mask_valid].reset_index(drop=True)

print(f"\n✓ Registros después de limpieza: {df_clean.shape[0]}")
print(f"✓ Valores faltantes restantes: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. CREAR CARACTERÍSTICAS (FEATURES) Y ETIQUETAS (LABELS)
# ============================================================================
print("\n[3] Ingeniería de características...")

# Columnas de entrada (features)
X = df_clean[sensor_cols].copy()

# Crear columna de diagnóstico: indicar si hay alguna falla
df_clean['hay_falla'] = (df_clean[falla_cols] > 0).any(axis=1).astype(int)

# Crear columna de tipo de falla (para análisis más detallado)
def clasificar_falla(row):
    """Clasifica el tipo de falla detectada"""
    if row['FRICCION LL_CHUMACERA B'] > 0 or row['FRICCION LA_CHUMACERA A'] > 0:
        return 'Fricción'
    elif row['DESBALANCE MTR_CHUMACERA B'] > 0 or row['DESBALANCE MTR_CHUMACERA A'] > 0:
        return 'Desbalance'
    elif row['DESALINEACION MTR-RED_CHUMACERA B'] > 0 or row['DESALINEACION MTR-RED_CHUMACERA A'] > 0:
        return 'Desalineación'
    elif row['IMPACTOS_CHUMACERA B'] > 0 or row['IMPACTOS_CHUMACERA A'] > 0:
        return 'Impactos'
    elif row['RODAMIENTO MTR LL_CHUMACERA B'] > 0 or row['RODAMIENTO MTR LA_CHUMACERA A'] > 0:
        return 'Rodamiento'
    else:
        return 'Normal'

df_clean['tipo_falla'] = df_clean[falla_cols].apply(
    lambda row: clasificar_falla(row), axis=1
)

# Etiqueta binaria (0=Normal, 1=Falla)
y = df_clean['hay_falla'].values

# Estadísticas
print(f"✓ Features (características): {X.shape[1]} columnas")
print(f"✓ Registros normales: {(y == 0).sum()}")
print(f"✓ Registros con falla: {(y == 1).sum()}")
print(f"✓ Distribución: {(y == 0).sum() / len(y) * 100:.1f}% Normal, {(y == 1).sum() / len(y) * 100:.1f}% Falla")

print("\nTipos de fallas detectadas:")
print(df_clean['tipo_falla'].value_counts())

# ============================================================================
# 4. NORMALIZAR DATOS
# ============================================================================
print("\n[4] Normalizando datos...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Datos normalizados (media=0, desv.est=1)")

# ============================================================================
# 5. DIVIDIR DATOS (TRAIN/TEST)
# ============================================================================
print("\n[5] Dividiendo datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Entrenamiento: {X_train.shape[0]} registros ({X_train.shape[0] / len(y) * 100:.1f}%)")
print(f"✓ Prueba: {X_test.shape[0]} registros ({X_test.shape[0] / len(y) * 100:.1f}%)")

# ============================================================================
# 6. ENTRENAR RANDOM FOREST
# ============================================================================
print("\n[6] Entrenando modelo Random Forest...")

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

modelo.fit(X_train, y_train)
print("✓ Modelo entrenado correctamente")

# Guardar modelo y escalador
joblib.dump(modelo, 'random_forest_model.joblib')
print("✓ Modelo Random Forest guardado: random_forest_model.joblib")
joblib.dump(scaler, 'scaler.joblib')
print("✓ Escalador guardado: scaler.joblib")

# ============================================================================
# 7. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n[7] Evaluando modelo...")

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas de entrenamiento
acc_train = accuracy_score(y_train, y_pred_train)
prec_train = precision_score(y_train, y_pred_train, zero_division=0)
rec_train = recall_score(y_train, y_pred_train, zero_division=0)
f1_train = f1_score(y_train, y_pred_train, zero_division=0)

# Métricas de prueba
acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test, zero_division=0)
rec_test = recall_score(y_test, y_pred_test, zero_division=0)
f1_test = f1_score(y_test, y_pred_test, zero_division=0)

print("\n📊 RESULTADOS DE ENTRENAMIENTO:")
print(f"  Exactitud:  {acc_train*100:.2f}%")
print(f"  Precisión:  {prec_train:.4f}")
print(f"  Recuperación: {rec_train:.4f}")
print(f"  F1-Score:   {f1_train:.4f}")

print("\n📊 RESULTADOS DE PRUEBA:")
print(f"  Exactitud:  {acc_test*100:.2f}%")
print(f"  Precisión:  {prec_test:.4f}")
print(f"  Recuperación: {rec_test:.4f}")
print(f"  F1-Score:   {f1_test:.4f}")

print("\n📋 REPORTE DE CLASIFICACIÓN (Test):")
print(classification_report(y_test, y_pred_test, 
      target_names=['Normal', 'Falla'],
      zero_division=0))

# ============================================================================
# 8. MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n[8] Matriz de confusión...")

cm = confusion_matrix(y_test, y_pred_test)
print(f"\nVerdaderos Negativos (TN):  {cm[0,0]}")
print(f"Falsos Positivos (FP):      {cm[0,1]}")
print(f"Falsos Negativos (FN):      {cm[1,0]}")
print(f"Verdaderos Positivos (TP):  {cm[1,1]}")

# ============================================================================
# 9. IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================
print("\n[9] Importancia de características...")

feature_importance = pd.DataFrame({
    'caracteristica': sensor_cols,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False)

print("\nTop 5 características más importantes:")
for i, row in feature_importance.head().iterrows():
    print(f"  {i+1}. {row['caracteristica']}: {row['importancia']:.4f}")

# ============================================================================
# 10. PREDICCIONES EN NUEVOS DATOS
# ============================================================================
print("\n[10] Predicciones en nuevos datos...")

# Tomar últimos 10 registros para predicción
ultimos_registros = df_clean.tail(10).copy()
X_nuevos = ultimos_registros[sensor_cols].values
X_nuevos_scaled = scaler.transform(X_nuevos)

predicciones = modelo.predict(X_nuevos_scaled)
probabilidades = modelo.predict_proba(X_nuevos_scaled)

print("\nÚltimas 10 predicciones:")
print("-" * 80)
for i, (idx, row) in enumerate(ultimos_registros.iterrows()):
    estado = "⚠️  FALLA" if predicciones[i] == 1 else "✓ NORMAL"
    prob = probabilidades[i][predicciones[i]] * 100
    print(f"{i+1}. {row['datetime']} | {estado} | Confianza: {prob:.1f}%")

# ============================================================================
# 11. GUARDAR PREDICCIONES
# ============================================================================
print("\n[11] Guardando resultados...")

# Agregar predicciones al dataframe
df_clean['prediccion'] = modelo.predict(X_scaled)
df_clean['confianza'] = modelo.predict_proba(X_scaled).max(axis=1)

# Guardar CSV con predicciones
df_clean.to_csv('Desfibradora_con_predicciones.csv', index=False)
print("✓ Archivo guardado: Desfibradora_con_predicciones.csv")

# Guardar resumen de características
feature_importance.to_csv('Importancia_caracteristicas.csv', index=False)
print("✓ Archivo guardado: Importancia_caracteristicas.csv")

# ============================================================================
# 12. VISUALIZACIONES
# ============================================================================
print("\n[12] Generando visualizaciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Desfibradora - Análisis de Predicción de Fallas', fontsize=16, fontweight='bold')

# 1. Matriz de confusión
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=['Normal', 'Falla'],
            yticklabels=['Normal', 'Falla'])
ax1.set_title('Matriz de Confusión (Test)')
ax1.set_ylabel('Real')
ax1.set_xlabel('Predicción')

# 2. Top 5 características
ax2 = axes[0, 1]
top_features = feature_importance.head()
ax2.barh(top_features['caracteristica'], top_features['importancia'], color='steelblue')
ax2.set_title('Top 5 Características Importantes')
ax2.set_xlabel('Importancia')

# 3. Distribución de clases
ax3 = axes[1, 0]
clases = ['Normal', 'Falla']
counts = [(y == 0).sum(), (y == 1).sum()]
ax3.bar(clases, counts, color=['green', 'red'], alpha=0.7)
ax3.set_title('Distribución de Clases en Dataset')
ax3.set_ylabel('Cantidad de registros')
for i, v in enumerate(counts):
    ax3.text(i, v + 10, str(v), ha='center', fontweight='bold')

# 4. Métricas de desempeño
ax4 = axes[1, 1]
metrics = ['Precisión', 'Exactitud', 'Recuperación', 'F1-Score']
train_vals = [acc_train, prec_train, rec_train, f1_train]
test_vals = [acc_test, prec_test, rec_test, f1_test]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, train_vals, width, label='Entrenamiento', alpha=0.8)
bars2 = ax4.bar(x + width/2, test_vals, width, label='Prueba', alpha=0.8)

ax4.set_ylabel('Valor')
ax4.set_title('Métricas de Desempeño')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=45, ha='right')
ax4.legend()
ax4.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('Analisis_prediccion_fallas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: Analisis_prediccion_fallas.png")
plt.close()

# ============================================================================
# 13. REPORTE FINAL
# ============================================================================
print("\n" + "="*80)
print("REPORTE FINAL")
print("="*80)

print(f"""
✓ DATASET PROCESADO
  • Registros iniciales: {df.shape[0]}
  • Registros después de limpieza: {df_clean.shape[0]}
  • Tasa de filtrado: {(1 - df_clean.shape[0]/df.shape[0])*100:.1f}%

✓ DISTRIBUCIÓN DE DATOS
  • Registros normales: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)
  • Registros con falla: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)
  • Desbalance: {max((y == 0).sum(), (y == 1).sum()) / min((y == 0).sum(), (y == 1).sum()):.1f}:1

✓ MODELO RANDOM FOREST
  • Árboles: 100
  • Profundidad máxima: 15
  • Criterio: Gini
  • Datos de entrenamiento: {X_train.shape[0]} ({X_train.shape[0] / len(y) * 100:.1f}%)
  • Datos de prueba: {X_test.shape[0]} ({X_test.shape[0] / len(y) * 100:.1f}%)

✓ DESEMPEÑO (Test Set)
  • Precisión general: {acc_test*100:.2f}%
  • Exactitud: {prec_test:.4f}
  • Recuperación: {rec_test:.4f}
  • F1-Score: {f1_test:.4f}

✓ PREDICCIONES
  • Registros predichos correctamente: {(y_test == y_pred_test).sum()} de {len(y_test)}
  • Fallos detectados (TP+FP): {(y_pred_test == 1).sum()}
  • Fallos no detectados (FN): {cm[1,0]}

✓ ARCHIVOS GENERADOS
  1. Desfibradora_con_predicciones.csv
  2. Importancia_caracteristicas.csv
  3. Analisis_prediccion_fallas.png
""")

print("="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
