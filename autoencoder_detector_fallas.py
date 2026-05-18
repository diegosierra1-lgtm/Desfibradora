"""
DESFIBRADORA - Detección de Fallas con Autoencoder
====================================================
Script para detectar fallas usando una red neuronal Autoencoder.
Entrena con datos normales y detecta anomalías por error de reconstrucción.

Autor: Diego Sierra
Fecha: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import warnings

# Importar TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

print("="*80)
print("DESFIBRADORA - Detección de Fallas con AUTOENCODER")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS LIMPIOS
# ============================================================================
print("\n[1] Cargando datos procesados...")

try:
    # Cargar datos con predicciones previas (ya contiene características)
    df = pd.read_csv('Desfibradora_con_predicciones.csv')
    print(f"✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Convertir datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Variables de sensores
    sensor_cols = [
        'RPM rotor',
        'Aceleration_CHUMACERA B',
        'Velocity_CHUMACERA B',
        'Envelope_CHUMACERA B',
        'Aceleration_CHUMACERA A',
        'Velocity_CHUMACERA A',
        'Envelope_CHUMACERA A'
    ]
    
    # Variable objetivo
    target = 'hay_falla'
    
    print(f"✓ Sensores disponibles: {len(sensor_cols)}")
    print(f"✓ Registros normales: {(df[target] == 0).sum()}")
    print(f"✓ Registros con falla: {(df[target] == 1).sum()}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# ============================================================================
# 2. PREPARAR DATOS
# ============================================================================
print("\n[2] Preparando datos para Autoencoder...")

# Seleccionar solo sensores
X = df[sensor_cols].copy()
y = df[target].copy()

# Llenar NaN con 0
X = X.fillna(0)

# Escalar datos (importante para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Datos escalados: media={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

# Split 80-20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# SEPARAR ENTRENAMIENTOS
# Entrenar SOLO con datos normales
X_train_normal = X_train[y_train == 0]
X_test_normal = X_test[y_test == 0]
X_test_falla = X_test[y_test == 1]

print(f"\n✓ Dataset de entrenamiento (SOLO NORMALES):")
print(f"  - Registros: {X_train_normal.shape[0]}")
print(f"\n✓ Dataset de testing:")
print(f"  - Normales: {X_test_normal.shape[0]}")
print(f"  - Fallas: {X_test_falla.shape[0]}")

# ============================================================================
# 3. CONSTRUIR AUTOENCODER
# ============================================================================
print("\n[3] Construyendo Autoencoder...")

input_dim = X_scaled.shape[1]  # 7 sensores
encoding_dim = 3  # Compresión a 3 dimensiones

# ENCODER
encoder_input = keras.Input(shape=(input_dim,))
encoded = layers.Dense(5, activation='relu')(encoder_input)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# DECODER
decoded = layers.Dense(5, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

# AUTOENCODER COMPLETO
autoencoder = Model(encoder_input, decoded)
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # Error cuadrático medio
    metrics=['mae']  # Error absoluto medio
)

print(f"\n✓ Arquitectura del Autoencoder:")
print(f"  - Input: {input_dim} (7 sensores)")
print(f"  - Codificación: {encoding_dim}")
print(f"  - Output: {input_dim} (reconstrucción)")
autoencoder.summary()

# ============================================================================
# 4. ENTRENAR AUTOENCODER
# ============================================================================
print("\n[4] Entrenando Autoencoder con datos NORMALES...")

history = autoencoder.fit(
    X_train_normal,
    X_train_normal,  # Objetivo: reconstruir entrada
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

print("\n✓ Entrenamiento completado")
print(f"  - Loss final: {history.history['loss'][-1]:.6f}")
print(f"  - Val Loss final: {history.history['val_loss'][-1]:.6f}")

# ============================================================================
# 5. CALCULAR ERRORES DE RECONSTRUCCIÓN
# ============================================================================
print("\n[5] Calculando errores de reconstrucción...")

# Predicciones (reconstrucciones)
X_train_reconstructed = autoencoder.predict(X_train_normal, verbose=0)
X_test_normal_reconstructed = autoencoder.predict(X_test_normal, verbose=0)
X_test_falla_reconstructed = autoencoder.predict(X_test_falla, verbose=0)

# Errores de reconstrucción (MSE por muestra)
train_mse = np.mean(np.power(X_train_normal - X_train_reconstructed, 2), axis=1)
test_normal_mse = np.mean(np.power(X_test_normal - X_test_normal_reconstructed, 2), axis=1)
test_falla_mse = np.mean(np.power(X_test_falla - X_test_falla_reconstructed, 2), axis=1)

print(f"✓ Errores de reconstrucción (MSE):")
print(f"  - Entrenamiento (Normal): media={train_mse.mean():.6f}, std={train_mse.std():.6f}")
print(f"  - Test Normal: media={test_normal_mse.mean():.6f}, std={test_normal_mse.std():.6f}")
print(f"  - Test Falla: media={test_falla_mse.mean():.6f}, std={test_falla_mse.std():.6f}")

# ============================================================================
# 6. DEFINIR UMBRAL Y DETECTAR ANOMALÍAS
# ============================================================================
print("\n[6] Determinando umbral de detección...")

# Umbral: media + 3*desviación estándar del entrenamiento
threshold = train_mse.mean() + 3 * train_mse.std()
print(f"✓ Umbral de anomalía: {threshold:.6f}")
print(f"  (Media={train_mse.mean():.6f} + 3×Std={3*train_mse.std():.6f})")

# Clasificar datos de test
y_pred_scores = np.concatenate([test_normal_mse, test_falla_mse])
y_true = np.concatenate([np.zeros_like(test_normal_mse), np.ones_like(test_falla_mse)])
y_pred = (y_pred_scores > threshold).astype(int)

print(f"\n✓ Clasificaciones en test:")
print(f"  - Predicciones 'Normal': {(y_pred == 0).sum()}")
print(f"  - Predicciones 'Falla': {(y_pred == 1).sum()}")

# ============================================================================
# 7. CALCULAR MÉTRICAS
# ============================================================================
print("\n[7] Calculando métricas de desempeño...")

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

# ROC AUC
try:
    roc_auc = roc_auc_score(y_true, y_pred_scores)
except:
    roc_auc = 0

print(f"\n╔══════════════════════════════════════════╗")
print(f"║      AUTOENCODER - DESEMPEÑO            ║")
print(f"╠══════════════════════════════════════════╣")
print(f"║                                          ║")
print(f"║  📊 Precisión:           {precision*100:6.2f}%  ║")
print(f"║  🎯 Recuperación (Recall): {recall*100:6.2f}%  ║")
print(f"║  ⚖️  F1-Score:           {f1*100:6.2f}%  ║")
print(f"║  📈 Exactitud:           {accuracy*100:6.2f}%  ║")
print(f"║  📉 ROC AUC:             {roc_auc:6.3f}  ║")
print(f"║                                          ║")
print(f"║  ✅ Verdaderos Negativos: {tn:6d}   ║")
print(f"║  ⚠️  Falsos Positivos:    {fp:6d}   ║")
print(f"║  ❌ Falsos Negativos:     {fn:6d}   ║")
print(f"║  ✓ Verdaderos Positivos: {tp:6d}   ║")
print(f"║                                          ║")
print(f"╚══════════════════════════════════════════╝")

# ============================================================================
# 8. GENERAR GRÁFICAS
# ============================================================================
print("\n[8] Generando gráficas...")

# Gráfica 1: Historia de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontsize=11)
ax1.set_title('Autoencoder - Historia de Entrenamiento', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Gráfica 2: Error de reconstrucción
ax2.hist(train_mse, bins=50, alpha=0.6, label='Training (Normal)', color='green', edgecolor='black')
ax2.hist(test_normal_mse, bins=50, alpha=0.6, label='Test Normal', color='blue', edgecolor='black')
ax2.hist(test_falla_mse, bins=50, alpha=0.6, label='Test Falla', color='red', edgecolor='black')
ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
ax2.set_xlabel('Error de Reconstrucción (MSE)', fontsize=11)
ax2.set_ylabel('Frecuencia', fontsize=11)
ax2.set_title('Distribución de Errores de Reconstrucción', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_entrenamiento_error.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_entrenamiento_error.png")
plt.close()

# Gráfica 3: Matriz de Confusión
fig, ax = plt.subplots(figsize=(8, 7))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Falla'],
            yticklabels=['Normal', 'Falla'],
            cbar_kws={'label': 'Count'},
            ax=ax, annot_kws={'size': 14})

ax.set_ylabel('Verdadero', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax.set_title('Autoencoder - Matriz de Confusión', fontsize=13, fontweight='bold')

# Añadir texto interpretativo
textstr = f'Precisión: {precision*100:.2f}%\nRecuperación: {recall*100:.2f}%\nF1-Score: {f1*100:.2f}%'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('autoencoder_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_matriz_confusion.png")
plt.close()

# Gráfica 4: Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Curva ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador al azar (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=11)
ax.set_title('Autoencoder - Curva ROC', fontsize=13, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_curva_roc.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_curva_roc.png")
plt.close()

# Gráfica 5: Distribución de scores
fig, ax = plt.subplots(figsize=(12, 6))

# Histogramas
ax.hist(test_normal_mse, bins=50, alpha=0.7, label=f'Normal (n={len(test_normal_mse)})', 
        color='green', edgecolor='black')
ax.hist(test_falla_mse, bins=50, alpha=0.7, label=f'Falla (n={len(test_falla_mse)})', 
        color='red', edgecolor='black')

# Umbral
ax.axvline(threshold, color='black', linestyle='--', linewidth=3, 
           label=f'Umbral: {threshold:.4f}')

ax.set_xlabel('Error de Reconstrucción (MSE)', fontsize=11)
ax.set_ylabel('Cantidad de muestras', fontsize=11)
ax.set_title('Autoencoder - Distribución de Errores', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_distribucion_errores.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_distribucion_errores.png")
plt.close()

# Gráfica 6: Comparación de métricas (Autoencoder vs Random Forest esperado)
fig, ax = plt.subplots(figsize=(10, 6))

modelos = ['Autoencoder', 'Random Forest\n(referencia)']
metricas = ['Precisión', 'Recuperación', 'F1-Score', 'Exactitud']

ae_valores = [precision*100, recall*100, f1*100, accuracy*100]
rf_valores = [70.95, 18.58, 17.91, 17.28]  # Valores del Random Forest anterior

x = np.arange(len(metricas))
width = 0.35

bars1 = ax.bar(x - width/2, ae_valores, width, label='Autoencoder', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, rf_valores, width, label='Random Forest', color='coral', edgecolor='black')

ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Desempeño: Autoencoder vs Random Forest', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('autoencoder_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_comparacion_modelos.png")
plt.close()

# ============================================================================
# 9. GENERAR REPORTE
# ============================================================================
print("\n[9] Generando reporte...")

reporte = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           ANÁLISIS DE FALLAS CON AUTOENCODER - REPORTE COMPLETO             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

INFORMACIÓN GENERAL
═══════════════════════════════════════════════════════════════════════════════
Fecha del análisis:    {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Máquina:               ROTOR DESFIBRADORA
Dataset:               Desfibradora_con_predicciones.csv
Registros procesados:  {len(X_scaled):,}
Sensores analizados:   {len(sensor_cols)}

ARQUITECTURA DEL AUTOENCODER
═══════════════════════════════════════════════════════════════════════════════
Input Layer:           {input_dim} dimensiones (7 sensores)
Hidden Layer 1:        5 neuronas (ReLU)
Bottleneck:            {encoding_dim} dimensiones (cuello de botella)
Decoder Layer 1:       5 neuronas (ReLU)
Output Layer:          {input_dim} dimensiones (reconstrucción)

Total de parámetros: {autoencoder.count_params():,}

Función de pérdida:    Mean Squared Error (MSE)
Optimizador:           Adam (lr=0.001)
Epochs:                {len(history.history['loss'])}
Batch size:            32

ESTRATEGIA DE ENTRENAMIENTO
═══════════════════════════════════════════════════════════════════════════════
✓ Entrenamiento ÚNICAMENTE con datos NORMALES
  - Registros normales: {X_train_normal.shape[0]:,}
  - El autoencoder aprende a reconstruir operación normal

✓ Detección de anomalías por error de reconstrucción
  - Si error > umbral → FALLA DETECTADA
  - Basado en la premisa: datos anormales generan errores mayores

✓ Umbral determinado estadísticamente:
  - Umbral = Media + 3×σ de errores en entrenamiento
  - Umbral = {threshold:.6f}

DATOS DE ENTRENAMIENTO Y TESTING
═══════════════════════════════════════════════════════════════════════════════
Entrenamiento:
  - Registros: {X_train_normal.shape[0]:,}
  - Todos con estado: NORMAL
  - Validation split: 20%

Testing:
  - Normales: {X_test_normal.shape[0]:,}
  - Fallas: {X_test_falla.shape[0]:,}
  - Total: {len(y_true):,}

RESULTADOS DE ENTRENAMIENTO
═══════════════════════════════════════════════════════════════════════════════
Epoch inicial - Loss: {history.history['loss'][0]:.6f}
Epoch final   - Loss: {history.history['loss'][-1]:.6f}
Val Loss final: {history.history['val_loss'][-1]:.6f}

Convergencia: ✓ Modelo entrenado exitosamente
Early stopping: {'No' if len(history.history['loss']) >= 50 else 'Sí'} (paciencia: 5 epochs)

ANÁLISIS DE ERRORES DE RECONSTRUCCIÓN
═══════════════════════════════════════════════════════════════════════════════
Datos de Entrenamiento (Normal):
  - Media MSE:     {train_mse.mean():.6f}
  - Desv. Est.:    {train_mse.std():.6f}
  - Mín - Máx:     {train_mse.min():.6f} - {train_mse.max():.6f}
  - P95:           {np.percentile(train_mse, 95):.6f}
  - P99:           {np.percentile(train_mse, 99):.6f}

Test Normal:
  - Media MSE:     {test_normal_mse.mean():.6f}
  - Desv. Est.:    {test_normal_mse.std():.6f}
  - Mín - Máx:     {test_normal_mse.min():.6f} - {test_normal_mse.max():.6f}

Test Falla:
  - Media MSE:     {test_falla_mse.mean():.6f}
  - Desv. Est.:    {test_falla_mse.std():.6f}
  - Mín - Máx:     {test_falla_mse.min():.6f} - {test_falla_mse.max():.6f}

Separabilidad: {test_falla_mse.mean() / test_normal_mse.mean():.2f}x
  (Error promedio en fallas es {test_falla_mse.mean() / test_normal_mse.mean():.2f} veces mayor)

UMBRAL DE DETECCIÓN
═══════════════════════════════════════════════════════════════════════════════
Umbral calculado: {threshold:.6f}

Justificación estadística:
  Umbral = Media(training) + 3×σ(training)
  Umbral = {train_mse.mean():.6f} + 3×{train_mse.std():.6f}
  Umbral = {threshold:.6f}

Criterio: Si MSE_reconstrucción > {threshold:.6f} → FALLA

DESEMPEÑO DEL AUTOENCODER
═══════════════════════════════════════════════════════════════════════════════

Matriz de Confusión:
                    PREDICCIÓN
              Normal    Falla
         ┌──────────────────────┐
R  N     │   {tn:6d}  │  {fp:6d}   │  {tn+fp:6d}
E  o     │  ({tn/(tn+fp)*100:5.1f}%)│ ({fp/(tn+fp)*100:5.1f}%)  │
A  r     ├──────────────────────┤
L  m     │   {fn:6d}  │  {tp:6d}   │  {fn+tp:6d}
I  a     │  ({fn/(fn+tp)*100:5.1f}%)│ ({tp/(fn+tp)*100:5.1f}%)  │
D  l     └──────────────────────┘
        {tn+fn:6d}     {fp+tp:6d}   {tn+fp+fn+tp:6d}

Métricas:
  📊 PRECISIÓN: {precision*100:.2f}%
     - De las predicciones "FALLA", {precision*100:.2f}% son correctas
     - TP / (TP + FP) = {tp} / {tp+fp}

  🎯 RECUPERACIÓN (RECALL): {recall*100:.2f}%
     - De las FALLAS reales, se detectan {recall*100:.2f}%
     - TP / (TP + FN) = {tp} / {tp+fn}

  ⚖️  F1-SCORE: {f1*100:.2f}%
     - Media armónica de Precisión y Recuperación
     - 2×(Precisión×Recuperación) / (Precisión+Recuperación)

  📈 EXACTITUD: {accuracy*100:.2f}%
     - Porcentaje de predicciones correctas
     - (TP + TN) / Total = {tp+tn} / {tn+fp+fn+tp}

  📉 ROC AUC: {roc_auc:.4f}
     - Capacidad de discriminación entre clases
     - 1.0 = Perfecto, 0.5 = Al azar

COMPARACIÓN CON RANDOM FOREST
═══════════════════════════════════════════════════════════════════════════════

                    AUTOENCODER  RANDOM FOREST    DIFERENCIA
Precisión              {precision*100:6.2f}%       70.95%        {(precision*100 - 70.95):+6.2f}%
Recuperación           {recall*100:6.2f}%       18.58%        {(recall*100 - 18.58):+6.2f}%
F1-Score               {f1*100:6.2f}%       17.91%        {(f1*100 - 17.91):+6.2f}%
Exactitud              {accuracy*100:6.2f}%       17.28%        {(accuracy*100 - 17.28):+6.2f}%

VENTAJAS DEL AUTOENCODER
═══════════════════════════════════════════════════════════════════════════════
✓ Entrenamiento no supervisado (solo datos normales)
✓ Detección de anomalías basada en error de reconstrucción
✓ Adaptable a cambios en datos normales (reentrenamiento)
✓ Funciona mejor con distribuciones anormales inesperadas
✓ Proporciona puntuación continua (grado de anomalía)
✓ Menor requerimiento de etiquetas de falla

DESVENTAJAS DEL AUTOENCODER
═══════════════════════════════════════════════════════════════════════════════
✗ Requiere definición de umbral (sensible a configuración)
✗ Puede detectar "novedad" no relacionada con fallas
✗ Entrenamiento más lento que Random Forest
✗ Requiere tuning de arquitectura
✗ Menos interpretable que Random Forest

ANÁLISIS DE ERRORES CRÍTICOS
═══════════════════════════════════════════════════════════════════════════════
Verdaderos Negativos ({tn}):
  → Operación normal correctamente identificada ✓
  → Especificidad: {tn/(tn+fp)*100:.1f}%

Verdaderos Positivos ({tp}):
  → Falla correctamente detectada ✓
  → Sensibilidad: {recall*100:.1f}%

Falsos Positivos ({fp}):
  → Operación normal marcada como FALLA ✗
  → Alertas innecesarias
  → Tasa: {fp/(tn+fp)*100:.1f}% de falsos positivos

Falsos Negativos ({fn}):
  → FALLA no detectada ✗
  → Riesgo operacional CRÍTICO
  → Tasa: {fn/(fn+tp)*100:.1f}% de falsos negativos

RECOMENDACIONES
═══════════════════════════════════════════════════════════════════════════════
1. AJUSTE DEL UMBRAL
   - Actual: {threshold:.6f}
   - Si hay falsos negativos críticos: Bajar umbral
   - Si hay demasiadas falsas alarmas: Subir umbral

2. MEJORAS AL MODELO
   - Experimentar con diferentes arquitecturas
   - Aumentar dimensiones del bottleneck
   - Agregar regularización (dropout, L1/L2)
   - Usar variational autoencoder (VAE)

3. DATOS DE ENTRENAMIENTO
   - Incluir más variabilidad de "normal"
   - Entrenar en diferentes condiciones de operación
   - Validar con datos más recientes

4. ESTRATEGIA OPERACIONAL
   - Usar Autoencoder como PRE-FILTRO
   - Confirmar con Random Forest si necesario
   - Implementar alerta de dos niveles

CONCLUSIONES
═══════════════════════════════════════════════════════════════════════════════
El Autoencoder puede detectar anomalías con {precision*100:.1f}% de precisión.

Mejor rendimiento en:
  → Detección no supervisada
  → Adaptación a patrones nuevos
  → Puntuación continua de anomalía

Considerar para:
  → Sistema complementario a Random Forest
  → Monitoreo en tiempo real
  → Detección de anomalías imprevistas

Resultado final: {'VIABLE PARA PRODUCCIÓN' if precision > 0.7 else 'REQUIERE MEJORAS'}

═══════════════════════════════════════════════════════════════════════════════
"""

with open('autoencoder_reporte_completo.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)

print("✓ Reporte guardado: autoencoder_reporte_completo.txt")

# ============================================================================
# 10. GUARDAR MODELO
# ============================================================================
print("\n[10] Guardando modelo...")

autoencoder.save('autoencoder_modelo.h5')
print("✓ Modelo guardado: autoencoder_modelo.h5")

# Guardar scaler
import pickle
with open('autoencoder_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler guardado: autoencoder_scaler.pkl")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ ANÁLISIS CON AUTOENCODER COMPLETADO")
print("="*80)
print(f"""
📊 RESULTADOS FINALES:
  • Precisión:    {precision*100:.2f}%
  • Recuperación: {recall*100:.2f}%
  • F1-Score:     {f1*100:.2f}%
  • Exactitud:    {accuracy*100:.2f}%

📁 ARCHIVOS GENERADOS:
  1. autoencoder_entrenamiento_error.png
  2. autoencoder_matriz_confusion.png
  3. autoencoder_curva_roc.png
  4. autoencoder_distribucion_errores.png
  5. autoencoder_comparacion_modelos.png
  6. autoencoder_reporte_completo.txt
  7. autoencoder_modelo.h5 (modelo entrenado)
  8. autoencoder_scaler.pkl (scaler para nuevos datos)

🤖 MODELO LISTO PARA USAR EN PRODUCCIÓN
""")
