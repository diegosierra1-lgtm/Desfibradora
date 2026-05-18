"""
DESFIBRADORA - Análisis de Tendencias con FILTRADO REAL
=========================================================
Compara datos CRUDOS (con ruido y outliers) vs FILTRADOS (outliers removidos + suavizado)
Muestra diferencias REALES en los gráficos aplicando técnicas de filtrado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.signal import savgol_filter, medfilt
from scipy.stats import iqr
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISIS DE TENDENCIAS: DATOS CRUDOS vs FILTRADOS")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1] Cargando datos...")

# Datos crudos
df_crudo = pd.read_csv('Desfibradora_crudo.csv')
df_crudo['datetime'] = pd.to_datetime(df_crudo['datetime'])
print(f"✓ Datos crudos cargados: {len(df_crudo)} registros")

# Los datos filtrados se generarán APLICANDO FILTROS REALES
print(f"✓ Se aplicarán filtros reales a los datos crudos")

# ============================================================================
# 2. DEFINIR VARIABLES
# ============================================================================
print("\n[2] Definiendo variables para análisis...")

sensor_cols = [
    'RPM rotor',
    'Aceleration_CHUMACERA B',
    'Velocity_CHUMACERA B',
    'Envelope_CHUMACERA B',
    'Aceleration_CHUMACERA A',
    'Velocity_CHUMACERA A',
    'Envelope_CHUMACERA A'
]

print(f"✓ Variables a analizar: {len(sensor_cols)}")
for i, col in enumerate(sensor_cols, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 3. FUNCIÓN PARA APLICAR FILTRADO REAL
# ============================================================================

def remover_outliers_iqr(datos):
    """Remueve outliers usando Rango Intercuartílico (IQR) - ignorando ceros"""
    # Copiar datos
    datos_limpio = datos.copy()
    
    # Trabajar solo con valores no-cero (datos reales)
    datos_no_cero = datos_limpio[datos_limpio != 0]
    
    if len(datos_no_cero) == 0:
        return datos_limpio, 0, 0, 0
    
    # Calcular Q1, Q3 e IQR usando solo datos no-cero
    Q1 = np.percentile(datos_no_cero, 25)
    Q3 = np.percentile(datos_no_cero, 75)
    IQR = Q3 - Q1
    
    # Definir límites (1.5 veces el IQR es estándar)
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Contar outliers ANTES de reemplazar
    outliers_mask = ((datos_limpio != 0) & 
                     ((datos_limpio < limite_inferior) | (datos_limpio > limite_superior)))
    
    # Reemplazar outliers con la mediana de valores válidos
    mediana = np.median(datos_no_cero)
    datos_limpio[outliers_mask] = mediana
    
    return datos_limpio, Q1, Q3, IQR

def suavizar_linea(datos, window_length=101):
    """Suaviza datos usando filtro Savitzky-Golay"""
    if len(datos) < window_length:
        window_length = len(datos) if len(datos) % 2 == 1 else len(datos) - 1
    
    if window_length < 3:
        return datos
    
    try:
        return savgol_filter(datos, window_length=window_length, polyorder=3)
    except:
        return datos

# ============================================================================
# 4. GENERAR GRÁFICOS COMPARATIVOS CON FILTRADO REAL
# ============================================================================
print("\n[3] Generando gráficos comparativos con filtrado REAL...")

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

for col in sensor_cols:
    print(f"\n  Procesando: {col}")
    
    # ========== OBTENER FECHAS ==========
    fechas = df_crudo['datetime'].values
    
    # ========== DATOS CRUDOS ==========
    crudo_datos = df_crudo[col].fillna(0).values.astype(float)
    crudo_indices = np.arange(len(crudo_datos))
    
    # ========== APLICAR FILTRADO REAL ==========
    # 1. Remover outliers usando IQR
    filtrado_datos, Q1, Q3, iqr_val = remover_outliers_iqr(crudo_datos)
    
    # 2. Aplicar filtro de mediana para suavizar más
    filtrado_datos = medfilt(filtrado_datos, kernel_size=5)
    
    # 3. Aplicar Savitzky-Golay para suavizado final
    filtrado_suavizado_base = suavizar_linea(filtrado_datos)
    
    filtrado_indices = np.arange(len(filtrado_datos))
    
    # ========== DETECTAR OUTLIERS USANDO IQR ==========
    datos_no_cero = crudo_datos[crudo_datos != 0]
    if len(datos_no_cero) > 0:
        Q1_calc = np.percentile(datos_no_cero, 25)
        Q3_calc = np.percentile(datos_no_cero, 75)
        IQR_calc = Q3_calc - Q1_calc
        limite_inf = Q1_calc - 1.5 * IQR_calc
        limite_sup = Q3_calc + 1.5 * IQR_calc
        outlier_mask = ((crudo_datos != 0) & 
                       ((crudo_datos < limite_inf) | (crudo_datos > limite_sup)))
        num_outliers = outlier_mask.sum()
    else:
        outlier_mask = np.zeros(len(crudo_datos), dtype=bool)
        num_outliers = 0
    
    # ========== ESTADÍSTICAS CRUDAS ==========
    crudo_stats = {
        'media': np.mean(crudo_datos),
        'std': np.std(crudo_datos),
        'min': np.min(crudo_datos),
        'max': np.max(crudo_datos),
        'ceros': (crudo_datos == 0).sum(),
        'total': len(crudo_datos),
        'outliers': num_outliers
    }
    
    # ========== ESTADÍSTICAS FILTRADAS ==========
    filtrado_stats = {
        'media': np.mean(filtrado_datos),
        'std': np.std(filtrado_datos),
        'min': np.min(filtrado_datos),
        'max': np.max(filtrado_datos),
        'ceros': (filtrado_datos == 0).sum(),
        'total': len(filtrado_datos),
        'outliers': 0  # Ya removidos
    }
    
    print(f"    Crudo: μ={crudo_stats['media']:.4f}, σ={crudo_stats['std']:.4f}, Outliers={crudo_stats['outliers']}")
    print(f"    Filtrado: μ={filtrado_stats['media']:.4f}, σ={filtrado_stats['std']:.4f}, ΔMedia={abs(crudo_stats['media']-filtrado_stats['media']):.4f}")
    
    # ========== CREAR GRÁFICO COMPARATIVO ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # ========== GRÁFICO 1: DATOS CRUDOS ==========
    ax1.scatter(fechas, crudo_datos, alpha=0.4, s=15, label='Datos crudos', color='red')
    
    # Suavizar datos crudos para la línea de tendencia
    crudo_suavizado = suavizar_linea(crudo_datos)
    ax1.plot(fechas, crudo_suavizado, 'r-', linewidth=3, label='Tendencia suavizada', zorder=10)
    
    # Línea de media
    ax1.axhline(y=crudo_stats['media'], color='darkred', linestyle='--', linewidth=2, 
                label=f"Media={crudo_stats['media']:.4f}", alpha=0.8)
    
    # Área de ±1 desviación estándar
    ax1.fill_between(fechas, 
                      crudo_stats['media'] - crudo_stats['std'],
                      crudo_stats['media'] + crudo_stats['std'],
                      alpha=0.2, color='red', label='±1 Desv. Est.')
    
    # Marcar outliers detectados
    if num_outliers > 0:
        ax1.scatter(fechas[outlier_mask], crudo_datos[outlier_mask], 
                   color='orange', s=50, marker='X', label=f'Outliers ({crudo_stats["outliers"]})', 
                   zorder=15, edgecolors='black', linewidths=1)
    
    ax1.set_title(f'{col} - DATOS CRUDOS (CON RUIDO Y OUTLIERS)\n' +
                  f'Registros: {crudo_stats["total"]:,} | Media: {crudo_stats["media"]:.4f} | ' +
                  f'Std: {crudo_stats["std"]:.4f} | Outliers: {crudo_stats["outliers"]}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Fecha y Hora', fontsize=11)
    ax1.set_ylabel(f'{col}', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== GRÁFICO 2: DATOS FILTRADOS ==========
    ax2.scatter(fechas, filtrado_datos, alpha=0.4, s=15, label='Datos filtrados', color='blue')
    
    # Suavizar datos filtrados
    ax2.plot(fechas, filtrado_suavizado_base, 'b-', linewidth=3, 
             label='Tendencia suavizada', zorder=10)
    
    # Línea de media
    ax2.axhline(y=filtrado_stats['media'], color='darkblue', linestyle='--', linewidth=2,
                label=f"Media={filtrado_stats['media']:.4f}", alpha=0.8)
    
    # Área de ±1 desviación estándar
    ax2.fill_between(fechas,
                      filtrado_stats['media'] - filtrado_stats['std'],
                      filtrado_stats['media'] + filtrado_stats['std'],
                      alpha=0.2, color='blue', label='±1 Desv. Est.')
    
    ax2.set_title(f'{col} - DATOS FILTRADOS (OUTLIERS REMOVIDOS + SUAVIZADO)\n' +
                  f'Registros: {filtrado_stats["total"]:,} | Media: {filtrado_stats["media"]:.4f} | ' +
                  f'Std: {filtrado_stats["std"]:.4f} | Cambio media: {abs(crudo_stats["media"]-filtrado_stats["media"]):.4f}',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_xlabel('Fecha y Hora', fontsize=11)
    ax2.set_ylabel(f'{col}', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Guardar figura
    nombre_archivo = col.replace(' ', '_').replace('_-_', '_').lower()
    ruta_imagen = f'tendencia_comparativa_{nombre_archivo}.png'
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    print(f"    ✓ Guardado: {ruta_imagen}")
    
    plt.close()

# ============================================================================
# 5. CREAR RESUMEN COMPARATIVO GENERAL
# ============================================================================
print("\n[4] Creando resumen comparativo...")

fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, 3*len(sensor_cols)))

if len(sensor_cols) == 1:
    axes = [axes]

fechas = df_crudo['datetime'].values

for idx, col in enumerate(sensor_cols):
    ax = axes[idx]
    
    # Datos crudos
    crudo_datos = df_crudo[col].fillna(0).values.astype(float)
    
    # Aplicar filtrado
    filtrado_datos, Q1, Q3, iqr_val = remover_outliers_iqr(crudo_datos)
    filtrado_datos = medfilt(filtrado_datos, kernel_size=5)
    
    # Suavizar
    crudo_suavizado = suavizar_linea(crudo_datos)
    filtrado_suavizado = suavizar_linea(filtrado_datos)
    
    # Graficar con fechas
    ax.plot(fechas, crudo_suavizado, label='Crudo (con ruido)', color='red', linewidth=2.5, alpha=0.8)
    ax.plot(fechas, filtrado_suavizado, label='Filtrado (sin outliers)', color='blue', linewidth=2.5, alpha=0.8)
    
    # Estadísticas
    crudo_media = np.mean(crudo_datos)
    filtrado_media = np.mean(filtrado_datos)
    
    ax.axhline(y=crudo_media, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(y=filtrado_media, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.5)
    
    diff_media = abs(crudo_media - filtrado_media)
    ax.set_title(f'{col} | Crudo: μ={crudo_media:.4f} | Filtrado: μ={filtrado_media:.4f} | Δμ={diff_media:.4f}',
                fontsize=12, fontweight='bold')
    ax.set_ylabel(col, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Formatear fechas en eje X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if idx == len(sensor_cols) - 1:
        ax.set_xlabel('Fecha y Hora', fontsize=10)

plt.tight_layout()
plt.savefig('tendencia_comparativa_resumen_total.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: tendencia_comparativa_resumen_total.png")
plt.close()

# ============================================================================
# 6. CREAR TABLA COMPARATIVA
# ============================================================================
print("\n[5] Generando tabla comparativa...")

datos_comparativa = []

for col in sensor_cols:
    crudo = df_crudo[col].fillna(0).values.astype(float)
    
    # Aplicar filtrado
    filtrado, Q1, Q3, iqr_val = remover_outliers_iqr(crudo)
    
    # Contar outliers removidos - solo en datos no-cero
    datos_no_cero = crudo[crudo != 0]
    if len(datos_no_cero) > 0:
        Q1_calc = np.percentile(datos_no_cero, 25)
        Q3_calc = np.percentile(datos_no_cero, 75)
        IQR_calc = Q3_calc - Q1_calc
        outliers_removidos = ((crudo != 0) & 
                             ((crudo < (Q1_calc - 1.5 * IQR_calc)) | 
                              (crudo > (Q3_calc + 1.5 * IQR_calc)))).sum()
    else:
        outliers_removidos = 0
    
    datos_comparativa.append({
        'Variable': col,
        'Crudo_Media': np.mean(crudo),
        'Crudo_Std': np.std(crudo),
        'Crudo_Min': np.min(crudo),
        'Crudo_Max': np.max(crudo),
        'Crudo_Ceros': (crudo == 0).sum(),
        'Crudo_Registros': len(crudo),
        'Filtrado_Media': np.mean(filtrado),
        'Filtrado_Std': np.std(filtrado),
        'Filtrado_Min': np.min(filtrado),
        'Filtrado_Max': np.max(filtrado),
        'Filtrado_Ceros': (filtrado == 0).sum(),
        'Outliers_Removidos': outliers_removidos,
        'Diff_Media': abs(np.mean(crudo) - np.mean(filtrado)),
        'Diff_Std': abs(np.std(crudo) - np.std(filtrado)),
        'Diff_Max': abs(np.max(crudo) - np.max(filtrado))
    })

df_comparativa = pd.DataFrame(datos_comparativa)
df_comparativa.to_csv('tendencia_comparativa_tabla.csv', index=False)
print("✓ Tabla guardada: tendencia_comparativa_tabla.csv")

# Mostrar resumen de la tabla
print("\n" + "="*80)
print("RESUMEN DE CAMBIOS POR VARIABLE")
print("="*80)
for _, row in df_comparativa.iterrows():
    print(f"\n{row['Variable']}:")
    print(f"  Media:      Crudo={row['Crudo_Media']:.6f} → Filtrado={row['Filtrado_Media']:.6f} (Δ={row['Diff_Media']:.6f})")
    print(f"  Desv. Est:  Crudo={row['Crudo_Std']:.6f} → Filtrado={row['Filtrado_Std']:.6f} (Δ={row['Diff_Std']:.6f})")
    print(f"  Rango:      Crudo=[{row['Crudo_Min']:.4f}, {row['Crudo_Max']:.4f}] → Filtrado=[{row['Filtrado_Min']:.4f}, {row['Filtrado_Max']:.4f}]")
    print(f"  Outliers removidos: {row['Outliers_Removidos']}")

# ============================================================================
# 7. GENERAR REPORTE
# ============================================================================
print("\n[6] Generando reporte...")

reporte = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         ANÁLISIS COMPARATIVO: DATOS CRUDOS vs FILTRADOS CON TÉCNICAS REALES  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

INFORMACIÓN GENERAL
═══════════════════════════════════════════════════════════════════════════════
Fecha del análisis:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Datos crudos:          {len(df_crudo):,} registros
Variables analizadas:  {len(sensor_cols)}
Técnicas de filtrado:  IQR (Rango Intercuartílico) + Mediana + Savitzky-Golay

TÉCNICAS DE FILTRADO APLICADAS
═══════════════════════════════════════════════════════════════════════════════
1. REMOVER OUTLIERS CON IQR:
   - Se calcula Q1 (percentil 25) y Q3 (percentil 75)
   - IQR = Q3 - Q1
   - Outliers: valores < (Q1 - 1.5*IQR) o > (Q3 + 1.5*IQR)
   - Los outliers se reemplazan con la mediana

2. FILTRO DE MEDIANA:
   - Suaviza datos removiendo picos espurios
   - Kernel size: 5 puntos

3. SAVITZKY-GOLAY:
   - Filtro polinómico (polyorder=3, window=101)
   - Preserva características importantes mientras suaviza

RESUMEN POR VARIABLE
═══════════════════════════════════════════════════════════════════════════════

"""

for idx, row in df_comparativa.iterrows():
    reporte += f"""
{idx+1}. {row['Variable'].upper()}
   ┌─ DATOS CRUDOS (CON RUIDO Y OUTLIERS)
   │  Media:        {row['Crudo_Media']:12.6f}  |  Desv. Est.: {row['Crudo_Std']:12.6f}
   │  Mín - Máx:    {row['Crudo_Min']:12.6f}  -  {row['Crudo_Max']:12.6f}
   │  Valores cero: {row['Crudo_Ceros']:,} de {row['Crudo_Registros']:,}
   │
   ├─ DATOS FILTRADOS (OUTLIERS REMOVIDOS + SUAVIZADO)
   │  Media:        {row['Filtrado_Media']:12.6f}  |  Desv. Est.: {row['Filtrado_Std']:12.6f}
   │  Mín - Máx:    {row['Filtrado_Min']:12.6f}  -  {row['Filtrado_Max']:12.6f}
   │  Valores cero: {row['Filtrado_Ceros']:,} (outliers removidos: {row['Outliers_Removidos']})
   │
   └─ IMPACTO DEL FILTRADO
      ΔMedia: {row['Diff_Media']:12.6f}  |  ΔStd: {row['Diff_Std']:12.6f}  |  ΔMax: {row['Diff_Max']:12.6f}
      Cambio media: {(row['Diff_Media']/row['Crudo_Media']*100 if row['Crudo_Media'] != 0 else 0):.2f}%
"""

reporte += """
INTERPRETACIÓN
═══════════════════════════════════════════════════════════════════════════════

✓ LÍNEAS DE TENDENCIA SUAVIZADAS:
  - Rojo (Crudo): Muestra el ruido y variabilidad del sensor original
  - Azul (Filtrado): Muestra la tendencia real sin outliers ni ruido

✓ OUTLIERS (marcados con X en naranja):
  - Valores atípicos detectados mediante IQR
  - Fueron reemplazados con la mediana para suavizar

✓ DESVIACIÓN ESTÁNDAR:
  - Crudo: Mayor valor = más ruido en los datos
  - Filtrado: Menor valor = datos más estables y limpios

✓ APLICACIONES PRÁCTICAS:
  - Los datos filtrados son más confiables para análisis de tendencias
  - Mejoran la precisión de modelos predictivos (Random Forest, etc.)
  - Permiten identificar patrones reales en lugar de ruido sensor

ARCHIVOS GENERADOS
═══════════════════════════════════════════════════════════════════════════════
✓ tendencia_comparativa_[variable].png - Gráficos detallados para cada sensor
✓ tendencia_comparativa_resumen_total.png - Comparación general de tendencias
✓ tendencia_comparativa_tabla.csv - Tabla con estadísticas comparativas
✓ tendencia_comparativa_reporte.txt - Este reporte

═══════════════════════════════════════════════════════════════════════════════
"""

# Guardar reporte
with open('tendencia_comparativa_reporte.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)

print("✓ Reporte guardado: tendencia_comparativa_reporte.txt")
print("\n" + "="*80)
print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
