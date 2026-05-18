"""
DESFIBRADORA - Monitor en Tiempo Real
======================================
Script que simula monitoreo en tiempo real con alertas

Ejecutar: python monitor_realtime.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
from datetime import datetime, timedelta
import random

class MonitorDesfibradora:
    """Monitor en tiempo real para la desfibradora"""
    
    def __init__(self):
        """Inicializa el monitor"""
        self.sensor_cols = [
            'RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
            'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 
            'Velocity_CHUMACERA A', 'Envelope_CHUMACERA A'
        ]
        
        # Cargar el modelo entrenado
        self.cargar_modelo()
        
        # Variables de monitoreo
        self.historial_predicciones = []
        self.alarmas_activas = []
        self.estadisticas = {
            'total_lecturas': 0,
            'fallos_detectados': 0,
            'ultimas_alertas': []
        }
    
    def cargar_modelo(self):
        """Carga el modelo entrenado"""
        try:
            df = pd.read_csv('Desfibradora_con_predicciones.csv')
            
            X = df[self.sensor_cols].fillna(0)
            
            # Entrenar scaler
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            
            # Crear modelo simple para demostración
            from sklearn.ensemble import RandomForestClassifier
            
            y = (df[[col for col in df.columns if 'CHUMACERA' in col and col not in self.sensor_cols]].sum(axis=1) > 0).astype(int)
            
            self.modelo = RandomForestClassifier(n_estimators=50, random_state=42)
            self.modelo.fit(self.scaler.transform(X), y)
            
            print("✓ Modelo cargado correctamente")
            
        except Exception as e:
            print(f"⚠️ Error cargando modelo: {e}")
            print("   Usando modelo simulado para demostración")
            self.modelo = None
            self.scaler = None
    
    def simular_lectura_sensor(self, falla=False):
        """Simula una lectura de sensor"""
        
        if falla:
            # Valores anormales
            return {
                'RPM rotor': np.random.uniform(500, 1500),
                'Aceleration_CHUMACERA B': np.random.uniform(1.0, 2.5),
                'Velocity_CHUMACERA B': np.random.uniform(3.0, 6.0),
                'Envelope_CHUMACERA B': np.random.uniform(2.0, 5.0),
                'Aceleration_CHUMACERA A': np.random.uniform(1.0, 2.5),
                'Velocity_CHUMACERA A': np.random.uniform(3.0, 6.0),
                'Envelope_CHUMACERA A': np.random.uniform(2.0, 5.0),
            }
        else:
            # Valores normales
            return {
                'RPM rotor': np.random.uniform(800, 1200),
                'Aceleration_CHUMACERA B': np.random.uniform(0.3, 0.6),
                'Velocity_CHUMACERA B': np.random.uniform(1.5, 2.8),
                'Envelope_CHUMACERA B': np.random.uniform(0.5, 2.0),
                'Aceleration_CHUMACERA A': np.random.uniform(0.3, 0.6),
                'Velocity_CHUMACERA A': np.random.uniform(1.5, 2.8),
                'Envelope_CHUMACERA A': np.random.uniform(0.5, 2.0),
            }
    
    def predecir(self, lectura):
        """Realiza predicción de la lectura"""
        
        X = np.array([[lectura[col] for col in self.sensor_cols]])
        
        if self.modelo and self.scaler:
            X_scaled = self.scaler.transform(X)
            prediccion = self.modelo.predict(X_scaled)[0]
            probabilidades = self.modelo.predict_proba(X_scaled)[0]
        else:
            # Simulación
            accel_promedio = np.mean([lectura['Aceleration_CHUMACERA B'], 
                                     lectura['Aceleration_CHUMACERA A']])
            prediccion = 1 if accel_promedio > 1.0 else 0
            probabilidades = [0.7, 0.3] if prediccion == 0 else [0.3, 0.7]
        
        confianza = max(probabilidades) * 100
        
        return {
            'prediccion': prediccion,
            'prob_normal': probabilidades[0] * 100,
            'prob_falla': probabilidades[1] * 100,
            'confianza': confianza,
            'estado': 'FALLA ⚠️' if prediccion == 1 else 'NORMAL ✓'
        }
    
    def procesar_lectura(self, timestamp, lectura, resultado):
        """Procesa una lectura con su predicción"""
        
        self.estadisticas['total_lecturas'] += 1
        
        registro = {
            'timestamp': timestamp,
            'lectura': lectura,
            'resultado': resultado,
            'momento': datetime.now()
        }
        
        self.historial_predicciones.append(registro)
        
        # Generar alertas
        if resultado['prediccion'] == 1:
            self.estadisticas['fallos_detectados'] += 1
            alerta = {
                'timestamp': timestamp,
                'tipo': 'FALLA_DETECTADA',
                'confianza': resultado['confianza'],
                'momentos_atras': datetime.now()
            }
            self.alarmas_activas.append(alerta)
            self.estadisticas['ultimas_alertas'].append(timestamp)
        
        return registro
    
    def mostrar_dashboard(self):
        """Muestra un dashboard en la terminal"""
        
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "DESFIBRADORA - MONITOR EN TIEMPO REAL".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        
        print(f"\n⏱️  Tiempo: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Lecturas totales: {self.estadisticas['total_lecturas']}")
        print(f"⚠️  Fallos detectados: {self.estadisticas['fallos_detectados']}")
        
        # Últimas lecturas
        print("\n" + "─" * 80)
        print("ÚLTIMAS 5 LECTURAS:")
        print("─" * 80)
        
        if len(self.historial_predicciones) > 0:
            for i, reg in enumerate(self.historial_predicciones[-5:]):
                ts = reg['timestamp']
                resultado = reg['resultado']
                print(f"{i+1}. {ts} │ {resultado['estado']} │ Confianza: {resultado['confianza']:6.1f}% │ P(Normal): {resultado['prob_normal']:5.1f}% │ P(Falla): {resultado['prob_falla']:5.1f}%")
        
        # Alertas activas
        print("\n" + "─" * 80)
        print("ALERTAS ACTIVAS:")
        print("─" * 80)
        
        if len(self.alarmas_activas) > 0:
            for i, alerta in enumerate(self.alarmas_activas[-5:]):
                print(f"🚨 {alerta['timestamp']} - Confianza: {alerta['confianza']:.1f}%")
        else:
            print("✓ Sin alertas activas")
        
        # Estadísticas
        print("\n" + "─" * 80)
        print("ESTADÍSTICAS:")
        print("─" * 80)
        
        if len(self.historial_predicciones) > 0:
            predicciones = [r['resultado']['prediccion'] for r in self.historial_predicciones]
            pct_normal = (predicciones.count(0) / len(predicciones)) * 100
            pct_falla = (predicciones.count(1) / len(predicciones)) * 100
            
            print(f"Estado NORMAL: {pct_normal:5.1f}%  ['{'█' * int(pct_normal/5)}{' ' * (20-int(pct_normal/5))}']")
            print(f"Estado FALLA:  {pct_falla:5.1f}%  ['{'█' * int(pct_falla/5)}{' ' * (20-int(pct_falla/5))}']")
        
        print("\n" + "─" * 80)
        print("Presiona Ctrl+C para salir")
    
    def iniciar_monitoreo(self, duracion=300, intervalo=5):
        """Inicia monitoreo continuo"""
        
        print("✓ Iniciando monitoreo...")
        print(f"  Duración: {duracion} segundos")
        print(f"  Intervalo: {intervalo} segundos")
        
        tiempo_inicio = time.time()
        secuencia_falla = []
        contador_lecturas = 0
        
        try:
            while time.time() - tiempo_inicio < duracion:
                contador_lecturas += 1
                
                # Decidir si simular falla (20% probabilidad)
                hay_falla = random.random() < 0.2
                
                # Simular lectura
                lectura = self.simular_lectura_sensor(falla=hay_falla)
                
                # Realizar predicción
                resultado = self.predecir(lectura)
                
                # Procesar
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.procesar_lectura(timestamp, lectura, resultado)
                
                # Mostrar dashboard
                self.mostrar_dashboard()
                
                # Esperar antes de siguiente lectura
                time.sleep(intervalo)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Monitoreo interrumpido por el usuario")
    
    def generar_reporte(self):
        """Genera reporte final"""
        
        print("\n" + "=" * 80)
        print("REPORTE DE MONITOREO".center(80))
        print("=" * 80)
        
        print(f"\n📊 RESUMEN")
        print(f"  • Total de lecturas: {self.estadisticas['total_lecturas']}")
        print(f"  • Fallos detectados: {self.estadisticas['fallos_detectados']}")
        print(f"  • Tasa de fallos: {(self.estadisticas['fallos_detectados'] / self.estadisticas['total_lecturas'] * 100):.1f}%")
        
        print(f"\n🚨 ALERTAS")
        print(f"  • Alertas totales: {len(self.alarmas_activas)}")
        
        if self.estadisticas['ultimas_alertas']:
            print(f"  • Última alerta: {self.estadisticas['ultimas_alertas'][-1]}")
        
        print(f"\n📈 DATOS HISTÓRICOS")
        print(f"  • Registros almacenados: {len(self.historial_predicciones)}")


def main():
    """Función principal"""
    
    monitor = MonitorDesfibradora()
    
    print("\n" + "=" * 80)
    print("DESFIBRADORA - MONITOR EN TIEMPO REAL".center(80))
    print("=" * 80)
    print("\nConfigurando monitor...")
    
    # Ejecutar monitoreo (5 minutos con lecturas cada 5 segundos)
    monitor.iniciar_monitoreo(duracion=60, intervalo=2)
    
    # Mostrar reporte final
    monitor.generar_reporte()


if __name__ == '__main__':
    main()
