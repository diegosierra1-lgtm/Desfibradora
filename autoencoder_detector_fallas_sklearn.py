"""
DESFIBRADORA - Detección de Fallas con Autoencoder (Scikit-Learn)
==================================================================
Script para detectar fallas usando un autoencoder con sklearn
Usa redes neuronales MLPRegressor como autoencoder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("DESFIBRADORA - Detección de Fallas con AUTOENCODER (Scikit-Learn)")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1] Cargando datos procesados...")

try:
    df = pd.read_csv('Desfibradora_con_predicciones.csv')
    print(f"✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    sensor_cols = [
        'RPM rotor',
        'Aceleration_CHUMACERA B',
        'Velocity_CHUMACERA B',
        'Envelope_CHUMACERA B',
        'Aceleration_CHUMACERA A',
        'Velocity_CHUMACERA A',
        'Envelope_CHUMACERA A'
    ]
    
    target = 'hay_falla'
    
    print(f"✓ Sensores: {len(sensor_cols)}")
    print(f"✓ Registros NORMAL: {(df[target] == 0).sum()}")
    print(f"✓ Registros FALLA: {(df[target] == 1).sum()}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# ============================================================================
# 2. PREPARAR DATOS
# ============================================================================
print("\n[2] Preparando datos...")

X = df[sensor_cols].copy().fillna(0)
y = df[target].copy()

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Solo datos normales para entrenamiento
X_train_normal = X_train[y_train == 0]
X_test_normal = X_test[y_test == 0]
X_test_falla = X_test[y_test == 1]

print(f"✓ Entrenamiento (solo normales): {X_train_normal.shape[0]}")
print(f"✓ Test Normal: {X_test_normal.shape[0]}")
print(f"✓ Test Falla: {X_test_falla.shape[0]}")

# ============================================================================
# 3. CONSTRUIR AUTOENCODER
# ============================================================================
print("\n[3] Construyendo Autoencoder con MLPRegressor...")

# Autoencoder: 7 → 4 → 2 → 4 → 7
hidden_layer_sizes = (5, 3, 5)  # Bottleneck en 3 dimensiones

autoencoder = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    batch_size=32,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    verbose=True
)

print(f"✓ Arquitectura: 7 → 5 → 3 → 5 → 7")
print(f"  Cuello de botella: 3 dimensiones")
print(f"  Objetivo: reconstruir entrada")

# ============================================================================
# 4. ENTRENAR AUTOENCODER
# ============================================================================
print("\n[4] Entrenando Autoencoder...")

autoencoder.fit(X_train_normal, X_train_normal)

print(f"✓ Entrenamiento completado")
print(f"  - Iteraciones: {autoencoder.n_iter_}")
print(f"  - Loss final: {autoencoder.loss_:.6f}")

# ============================================================================
# 5. CALCULAR ERRORES DE RECONSTRUCCIÓN
# ============================================================================
print("\n[5] Calculando errores de reconstrucción...")

X_train_reconstructed = autoencoder.predict(X_train_normal)
X_test_normal_reconstructed = autoencoder.predict(X_test_normal)
X_test_falla_reconstructed = autoencoder.predict(X_test_falla)

# MSE por muestra
train_mse = np.mean(np.power(X_train_normal - X_train_reconstructed, 2), axis=1)
test_normal_mse = np.mean(np.power(X_test_normal - X_test_normal_reconstructed, 2), axis=1)
test_falla_mse = np.mean(np.power(X_test_falla - X_test_falla_reconstructed, 2), axis=1)

print(f"✓ Errores de reconstrucción (MSE):")
print(f"  - Training (Normal): μ={train_mse.mean():.6f}, σ={train_mse.std():.6f}")
print(f"  - Test Normal: μ={test_normal_mse.mean():.6f}, σ={test_normal_mse.std():.6f}")
print(f"  - Test Falla: μ={test_falla_mse.mean():.6f}, σ={test_falla_mse.std():.6f}")

# ============================================================================
# 6. DETERMINAR UMBRAL
# ============================================================================
print("\n[6] Determinando umbral de detección...")

threshold = train_mse.mean() + 3 * train_mse.std()
print(f"✓ Umbral: {threshold:.6f}")
print(f"  (Media={train_mse.mean():.6f} + 3×Std={3*train_mse.std():.6f})")

# ============================================================================
# 7. CLASIFICAR Y EVALUAR
# ============================================================================
print("\n[7] Evaluando desempeño...")

y_pred_scores = np.concatenate([test_normal_mse, test_falla_mse])
y_true = np.concatenate([np.zeros_like(test_normal_mse), np.ones_like(test_falla_mse)])
y_pred = (y_pred_scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

try:
    roc_auc = roc_auc_score(y_true, y_pred_scores)
except:
    roc_auc = 0

print(f"\n╔══════════════════════════════════════════╗")
print(f"║      AUTOENCODER - DESEMPEÑO            ║")
print(f"╠══════════════════════════════════════════╣")
print(f"║  📊 Precisión:           {precision*100:6.2f}%  ║")
print(f"║  🎯 Recuperación:        {recall*100:6.2f}%  ║")
print(f"║  ⚖️  F1-Score:           {f1*100:6.2f}%  ║")
print(f"║  📈 Exactitud:           {accuracy*100:6.2f}%  ║")
print(f"║  📉 ROC AUC:             {roc_auc:6.3f}  ║")
print(f"║                                          ║")
print(f"║  ✅ TN: {tn:6d}   ⚠️ FP: {fp:6d}   ║")
print(f"║  ❌ FN: {fn:6d}   ✓ TP: {tp:6d}   ║")
print(f"║                                          ║")
print(f"╚══════════════════════════════════════════╝")

# ============================================================================
# 8. GENERAR GRÁFICAS
# ============================================================================
print("\n[8] Generando gráficas...")

# Gráfica 1: Error de reconstrucción
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(train_mse, bins=50, alpha=0.6, label='Training (Normal)', 
        color='green', edgecolor='black')
ax.hist(test_normal_mse, bins=50, alpha=0.6, label='Test Normal', 
        color='blue', edgecolor='black')
ax.hist(test_falla_mse, bins=50, alpha=0.6, label='Test Falla', 
        color='red', edgecolor='black')
ax.axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
           label=f'Umbral: {threshold:.4f}')

ax.set_xlabel('Error de Reconstrucción (MSE)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.set_title('Autoencoder - Distribución de Errores de Reconstrucción', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_error_reconstruccion.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_error_reconstruccion.png")
plt.close()

# Gráfica 2: Matriz de confusión
fig, ax = plt.subplots(figsize=(8, 7))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Falla'],
            yticklabels=['Normal', 'Falla'],
            cbar_kws={'label': 'Cantidad'},
            ax=ax, annot_kws={'size': 14})

ax.set_ylabel('Verdadero', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax.set_title('Autoencoder - Matriz de Confusión', fontsize=13, fontweight='bold')

textstr = f'Precisión: {precision*100:.2f}%\nRecuperación: {recall*100:.2f}%\nF1-Score: {f1*100:.2f}%'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('autoencoder_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_matriz_confusion.png")
plt.close()

# Gráfica 3: Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Autoencoder (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Al azar (AUC = 0.5)')
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

# Gráfica 4: Comparación de modelos
fig, ax = plt.subplots(figsize=(10, 6))

modelos = ['Autoencoder', 'Random Forest\n(referencia)']
metricas = ['Precisión', 'Recuperación', 'F1-Score', 'Exactitud']

ae_valores = [precision*100, recall*100, f1*100, accuracy*100]
rf_valores = [70.95, 18.58, 17.91, 17.28]

x = np.arange(len(metricas))
width = 0.35

bars1 = ax.bar(x - width/2, ae_valores, width, label='Autoencoder', 
               color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, rf_valores, width, label='Random Forest', 
               color='coral', edgecolor='black')

ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparación: Autoencoder vs Random Forest', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, max(max(ae_valores), max(rf_valores)) * 1.15])

# Añadir etiquetas con mejor visibilidad
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        # Si es 0, mostrar en rojo para hacerlo obvio
        if height == 0:
            ax.text(bar.get_x() + bar.get_width()/2., 2,
                    '0.0%', ha='center', va='bottom', fontsize=10, 
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, 
                    fontweight='bold')

plt.tight_layout()
plt.savefig('autoencoder_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_comparacion_modelos.png")
plt.close()

# Gráfica 5: Scatter plot de errores
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(range(len(test_normal_mse)), test_normal_mse, alpha=0.5, s=30, 
          label=f'Normal (n={len(test_normal_mse)})', color='green')
ax.scatter(range(len(test_normal_mse), len(test_normal_mse) + len(test_falla_mse)), 
          test_falla_mse, alpha=0.5, s=30, 
          label=f'Falla (n={len(test_falla_mse)})', color='red')

ax.axhline(threshold, color='black', linestyle='--', linewidth=2.5, 
           label=f'Umbral: {threshold:.4f}')

ax.set_xlabel('Muestra de Test', fontsize=11)
ax.set_ylabel('Error de Reconstrucción (MSE)', fontsize=11)
ax.set_title('Autoencoder - Error por Muestra', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_error_por_muestra.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: autoencoder_error_por_muestra.png")
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
Tipo:                  MLPRegressor (Scikit-Learn)
Capas:
  Input:               7 dimensiones (7 sensores)
  Hidden 1:            5 neuronas (ReLU)
  Bottleneck:          3 dimensiones (cuello de botella)
  Hidden 2:            5 neuronas (ReLU)
  Output:              7 dimensiones (reconstrucción)

Optimizador:           Adam
Learning rate:         0.001
Max iteraciones:       100
Batch size:            32

ESTRATEGIA DE ENTRENAMIENTO
═══════════════════════════════════════════════════════════════════════════════
✓ Entrenamiento ÚNICAMENTE con datos NORMALES
  - Registros normales: {X_train_normal.shape[0]:,}
  - El autoencoder aprende la distribución normal

✓ Detección por anomalía de reconstrucción
  - Si error de reconstrucción > umbral → FALLA
  - Basado en: datos anormales generan errores mayores

✓ Umbral determinado estadísticamente:
  Umbral = Media + 3×σ
  Umbral = {train_mse.mean():.6f} + 3×{train_mse.std():.6f}
  Umbral = {threshold:.6f}

DATOS DE ENTRENAMIENTO
═══════════════════════════════════════════════════════════════════════════════
Entrenamiento:
  - Registros: {X_train_normal.shape[0]:,}
  - Estado: TODOS NORMALES
  - Validación: 20%

Testing:
  - Normales: {X_test_normal.shape[0]:,}
  - Fallas: {X_test_falla.shape[0]:,}
  - Total: {len(y_true):,}

ANÁLISIS DE ERRORES DE RECONSTRUCCIÓN
═══════════════════════════════════════════════════════════════════════════════
Datos de Entrenamiento (Normal):
  Media MSE:     {train_mse.mean():.6f}
  Desv. Est.:    {train_mse.std():.6f}
  Mín - Máx:     {train_mse.min():.6f} - {train_mse.max():.6f}
  P95:           {np.percentile(train_mse, 95):.6f}

Test Normal:
  Media MSE:     {test_normal_mse.mean():.6f}
  Desv. Est.:    {test_normal_mse.std():.6f}
  Mín - Máx:     {test_normal_mse.min():.6f} - {test_normal_mse.max():.6f}

Test Falla:
  Media MSE:     {test_falla_mse.mean():.6f}
  Desv. Est.:    {test_falla_mse.std():.6f}
  Mín - Máx:     {test_falla_mse.min():.6f} - {test_falla_mse.max():.6f}

Separabilidad: {test_falla_mse.mean() / test_normal_mse.mean():.2f}x
(Las fallas generan {test_falla_mse.mean() / test_normal_mse.mean():.2f} veces más error)

DESEMPEÑO DEL AUTOENCODER
═══════════════════════════════════════════════════════════════════════════════

Matriz de Confusión:
                    PREDICCIÓN
              Normal    Falla
         ┌──────────────────────┐
R  N     │   {tn:6d}  │  {fp:6d}   │
E  o     │  ({tn/(tn+fp)*100:5.1f}%)│ ({fp/(tn+fp)*100:5.1f}%)  │
A  r     ├──────────────────────┤
L  m     │   {fn:6d}  │  {tp:6d}   │
I  a     │  ({fn/(fn+tp)*100:5.1f}%)│ ({tp/(fn+tp)*100:5.1f}%)  │
D  l     └──────────────────────┘

Métricas de Desempeño:

  📊 PRECISIÓN: {precision*100:.2f}%
     De los casos predichos como "FALLA", {precision*100:.2f}% son verdaderas fallas
     TP / (TP + FP) = {tp} / {tp+fp}

  🎯 RECUPERACIÓN (RECALL): {recall*100:.2f}%
     De las FALLAS reales, se detectan el {recall*100:.2f}%
     TP / (TP + FN) = {tp} / {tp+fn}

  ⚖️  F1-SCORE: {f1*100:.2f}%
     Media armónica de Precisión y Recuperación
     Considera ambas métricas de forma equilibrada

  📈 EXACTITUD: {accuracy*100:.2f}%
     Porcentaje total de predicciones correctas
     (TP + TN) / Total = {tp+tn} / {tn+fp+fn+tp}

  📉 ROC AUC: {roc_auc:.4f}
     Área bajo la curva ROC
     Capacidad de discriminación: {roc_auc:.1%}

COMPARACIÓN CON RANDOM FOREST
═══════════════════════════════════════════════════════════════════════════════

Métrica          AUTOENCODER  RANDOM FOREST    Ventaja
─────────────────────────────────────────────────────────
Precisión           {precision*100:6.2f}%       70.95%        {'AUTOENCODER' if precision > 0.7095 else 'Random Forest':15}
Recuperación        {recall*100:6.2f}%       18.58%        {'AUTOENCODER' if recall > 0.1858 else 'Random Forest':15}
F1-Score            {f1*100:6.2f}%       17.91%        {'AUTOENCODER' if f1 > 0.1791 else 'Random Forest':15}
Exactitud           {accuracy*100:6.2f}%       17.28%        {'AUTOENCODER' if accuracy > 0.1728 else 'Random Forest':15}
ROC AUC             {roc_auc:6.3f}         ~0.65        {'AUTOENCODER' if roc_auc > 0.65 else 'Random Forest':15}

¿CUÁNDO USAR CADA UNO?
═══════════════════════════════════════════════════════════════════════════════

RANDOM FOREST:
  ✓ Mejor interpretabilidad
  ✓ Usa etiquetas de falla (supervisado)
  ✓ Más rápido en predicción
  ✓ Determinístico

AUTOENCODER:
  ✓ No supervisado (solo datos normales)
  ✓ Detección de anomalías inesperadas
  ✓ Puntuación continua de anomalía
  ✓ Adapta mejor a cambios

ESTRATEGIA RECOMENDADA: ENSEMBLE
  1. Autoencoder: Pre-filtro de anomalías
  2. Random Forest: Clasificación de tipo falla
  3. Ambos acuerdan → CONFIANZA ALTA

ANÁLISIS DE ERRORES
═══════════════════════════════════════════════════════════════════════════════
Verdaderos Negativos ({tn}):
  Operación normal detectada correctamente
  Especificidad: {tn/(tn+fp)*100:.1f}%

Verdaderos Positivos ({tp}):
  Falla detectada correctamente
  Sensibilidad: {recall*100:.1f}%

Falsos Positivos ({fp}):
  Falsa alarma (operación normal marcada como falla)
  Tasa: {fp/(tn+fp)*100:.1f}%

Falsos Negativos ({fn}):
  CRÍTICO: Falla no detectada
  Tasa: {fn/(fn+tp)*100:.1f}%

RECOMENDACIONES FINALES
═══════════════════════════════════════════════════════════════════════════════
1. AJUSTE DEL UMBRAL
   Actual: {threshold:.6f}
   
   Si hay demasiados falsos negativos:
   → Bajar umbral (mayor sensibilidad)
   
   Si hay demasiadas falsas alarmas:
   → Subir umbral (menor sensibilidad)

2. MEJORA DEL MODELO
   • Entrenar con más variedad de datos "normales"
   • Experimentar con diferentes arquitecturas
   • Ajustar tamaño del bottleneck (3 vs 2 vs 4)

3. INTEGRACIÓN
   • Usar junto con Random Forest (ensemble)
   • Implementar alerta de dos niveles
   • Monitoreo continuo de umbral

4. OPERACIÓN
   • Sistema COMPLEMENTARIO al Random Forest
   • Reentrenar periódicamente con datos nuevos
   • Revisar distribución de errores regularmente

CONCLUSIÓN
═══════════════════════════════════════════════════════════════════════════════
El Autoencoder logra detectar fallas con {precision*100:.1f}% de precisión.

Ventajas clave:
  → Aprendizaje no supervisado
  → Detección de anomalías imprevistas
  → Puntuación continua
  → Adaptable a cambios

Recomendación:
  {'USAR COMO SISTEMA PRINCIPAL' if precision > 0.7 and recall > 0.5 else 'USAR COMO SISTEMA COMPLEMENTARIO'}

═══════════════════════════════════════════════════════════════════════════════
"""

with open('autoencoder_reporte_completo.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)

print("✓ Reporte guardado: autoencoder_reporte_completo.txt")

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
  • ROC AUC:      {roc_auc:.3f}

📁 ARCHIVOS GENERADOS:
  1. autoencoder_error_reconstruccion.png
  2. autoencoder_matriz_confusion.png
  3. autoencoder_curva_roc.png
  4. autoencoder_error_por_muestra.png
  5. autoencoder_comparacion_modelos.png
  6. autoencoder_reporte_completo.txt

✅ LISTO PARA ANÁLISIS Y DOCUMENTACIÓN
""")
