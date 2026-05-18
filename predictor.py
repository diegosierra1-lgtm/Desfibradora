"""
DESFIBRADORA - Script de Predicción en Tiempo Real
====================================================
Carga el modelo entrenado y permite hacer predicciones en nuevos datos.

Uso:
  python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 --env_b 1.6 \
                      --accel_a 0.54 --vel_a 1.7 --env_a 0.99
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import argparse
import json

class PredictorDesfibradora:
    """Predictor de fallas para la desfibradora"""
    
    def __init__(self, model_path='random_forest_model.joblib', 
                 scaler_path='scaler.joblib'):
        """Carga el modelo y scaler"""
        self.modelo = None
        self.scaler = None
        
        try:
            # Cargar modelo y scaler entrenados
            if model_path.endswith('.joblib'):
                self.modelo = joblib.load(model_path)
            if scaler_path.endswith('.joblib'):
                self.scaler = joblib.load(scaler_path)
            
            if self.modelo is None or self.scaler is None:
                self.cargar_modelo_desde_archivo()
        except:
            self.cargar_modelo_desde_archivo()
    
    def cargar_modelo_desde_archivo(self):
        """Intenta cargar el modelo del CSV guardado"""
        try:
            # Intentar cargar desde archivos .joblib primero
            try:
                self.modelo = joblib.load('random_forest_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                print("✓ Modelo cargado desde archivos .joblib")
                return
            except:
                pass
            
            # Si no existen archivos .joblib, entrenar desde el CSV
            df = pd.read_csv('Desfibradora_con_predicciones.csv')
            
            sensor_cols = [
                'RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
                'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 
                'Velocity_CHUMACERA A', 'Envelope_CHUMACERA A'
            ]
            
            X = df[sensor_cols].fillna(0)
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            
            print("⚠️  Nota: Usa primero el script principal (analisis_prediccion_fallas.py) para entrenar el modelo")
            
        except Exception as e:
            print(f"Error: {e}")
            self.modelo = None
            self.scaler = None
    
    def predecir_simple(self, rpm, accel_b, vel_b, env_b, accel_a, vel_a, env_a):
        """
        Hace predicción con valores individuales
        
        Parámetros:
        -----------
        rpm : float - RPM del rotor
        accel_b : float - Aceleración CHUMACERA B
        vel_b : float - Velocidad CHUMACERA B
        env_b : float - Envelope CHUMACERA B
        accel_a : float - Aceleración CHUMACERA A
        vel_a : float - Velocidad CHUMACERA A
        env_a : float - Envelope CHUMACERA A
        """
        
        if self.modelo is None or self.scaler is None:
            print("❌ Modelo no cargado correctamente")
            return None
        
        # Obtener características temporales del momento actual
        from datetime import datetime
        now = datetime.now()
        hora = now.hour
        minuto = now.minute
        dia_semana = now.weekday()
        dia_mes = now.day
        mes = now.month
        
        # Generar características temporales normalizadas
        hora_sin = np.sin(2 * np.pi * hora / 24)
        hora_cos = np.cos(2 * np.pi * hora / 24)
        minuto_sin = np.sin(2 * np.pi * minuto / 60)
        minuto_cos = np.cos(2 * np.pi * minuto / 60)
        dia_semana_sin = np.sin(2 * np.pi * dia_semana / 7)
        dia_semana_cos = np.cos(2 * np.pi * dia_semana / 7)
        dia_mes_normalizado = (dia_mes - 1) / 30
        mes_normalizado = (mes - 1) / 11
        
        # Crear vector de entrada con todas las características (7 sensores + 8 temporales)
        X = np.array([[
            rpm, accel_b, vel_b, env_b, accel_a, vel_a, env_a,
            hora_sin, hora_cos, minuto_sin, minuto_cos,
            dia_semana_sin, dia_semana_cos, dia_mes_normalizado, mes_normalizado
        ]])
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        prediccion = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'estado': 'FALLA ⚠️' if prediccion == 1 else 'NORMAL ✓',
            'prediccion': prediccion,
            'prob_normal': probabilidades[0] * 100,
            'prob_falla': probabilidades[1] * 100,
            'confianza': max(probabilidades) * 100
        }
    
    def predecir_desde_csv(self, ruta_csv):
        """Predice fallas para nuevos datos en CSV"""
        
        try:
            df = pd.read_csv(ruta_csv)
            
            sensor_cols = [
                'RPM rotor', 'Aceleration_CHUMACERA B', 'Velocity_CHUMACERA B',
                'Envelope_CHUMACERA B', 'Aceleration_CHUMACERA A', 
                'Velocity_CHUMACERA A', 'Envelope_CHUMACERA A'
            ]
            
            # Validar columnas
            missing_cols = [col for col in sensor_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Columnas faltantes: {missing_cols}")
                return None
            
            X = df[sensor_cols].fillna(0).copy()
            
            # Generar características temporales
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                hora = df['datetime'].dt.hour
                minuto = df['datetime'].dt.minute
                dia_semana = df['datetime'].dt.dayofweek
                dia_mes = df['datetime'].dt.day
                mes = df['datetime'].dt.month
            else:
                from datetime import datetime
                now = datetime.now()
                hora = np.full(len(df), now.hour)
                minuto = np.full(len(df), now.minute)
                dia_semana = np.full(len(df), now.weekday())
                dia_mes = np.full(len(df), now.day)
                mes = np.full(len(df), now.month)
            
            # Agregar características temporales
            X['hora_sin'] = np.sin(2 * np.pi * hora / 24)
            X['hora_cos'] = np.cos(2 * np.pi * hora / 24)
            X['minuto_sin'] = np.sin(2 * np.pi * minuto / 60)
            X['minuto_cos'] = np.cos(2 * np.pi * minuto / 60)
            X['dia_semana_sin'] = np.sin(2 * np.pi * dia_semana / 7)
            X['dia_semana_cos'] = np.cos(2 * np.pi * dia_semana / 7)
            X['dia_mes_normalizado'] = (dia_mes - 1) / 30
            X['mes_normalizado'] = (mes - 1) / 11
            
            # Reordenar columnas para que coincidan con el entrenamiento
            all_cols = sensor_cols + [
                'hora_sin', 'hora_cos', 'minuto_sin', 'minuto_cos',
                'dia_semana_sin', 'dia_semana_cos', 'dia_mes_normalizado', 'mes_normalizado'
            ]
            X = X[all_cols]
            
            X_scaled = self.scaler.transform(X)
            
            predicciones = self.modelo.predict(X_scaled)
            probabilidades = self.modelo.predict_proba(X_scaled)
            
            # Agregar predicciones al dataframe
            df['prediccion'] = predicciones
            df['prob_normal'] = probabilidades[:, 0] * 100
            df['prob_falla'] = probabilidades[:, 1] * 100
            df['confianza'] = np.max(probabilidades, axis=1) * 100
            df['estado'] = df['prediccion'].apply(
                lambda x: 'FALLA ⚠️' if x == 1 else 'NORMAL ✓'
            )
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(
        description='Predictor de fallas para la Desfibradora'
    )
    
    parser.add_argument('--rpm', type=float, help='RPM del rotor')
    parser.add_argument('--accel_b', type=float, help='Aceleración CHUMACERA B')
    parser.add_argument('--vel_b', type=float, help='Velocidad CHUMACERA B')
    parser.add_argument('--env_b', type=float, help='Envelope CHUMACERA B')
    parser.add_argument('--accel_a', type=float, help='Aceleración CHUMACERA A')
    parser.add_argument('--vel_a', type=float, help='Velocidad CHUMACERA A')
    parser.add_argument('--env_a', type=float, help='Envelope CHUMACERA A')
    parser.add_argument('--csv', type=str, help='Ruta a archivo CSV para predicción')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DESFIBRADORA - Predictor de Fallas")
    print("="*70)
    
    predictor = PredictorDesfibradora()
    
    # Predicción desde valores individuales
    if args.rpm is not None:
        print("\n📊 Predicción de valores individuales:")
        
        resultado = predictor.predecir_simple(
            rpm=args.rpm or 0,
            accel_b=args.accel_b or 0,
            vel_b=args.vel_b or 0,
            env_b=args.env_b or 0,
            accel_a=args.accel_a or 0,
            vel_a=args.vel_a or 0,
            env_a=args.env_a or 0
        )
        
        if resultado:
            print(f"\n✓ Estado: {resultado['estado']}")
            print(f"  Probabilidad Normal: {resultado['prob_normal']:.2f}%")
            print(f"  Probabilidad Falla:  {resultado['prob_falla']:.2f}%")
            print(f"  Confianza:          {resultado['confianza']:.2f}%")
    
    # Predicción desde CSV
    elif args.csv:
        print(f"\n📂 Leyendo CSV: {args.csv}")
        
        resultado = predictor.predecir_desde_csv(args.csv)
        
        if resultado is not None:
            # Mostrar resumen
            print(f"\n✓ Predicciones completadas para {len(resultado)} registros")
            print(f"\nResumen:")
            print(f"  - Estado NORMAL:  {(resultado['prediccion'] == 0).sum()}")
            print(f"  - Estado FALLA:   {(resultado['prediccion'] == 1).sum()}")
            print(f"\nPrimeros 10 registros con predicción:")
            print(resultado[['datetime', 'estado', 'confianza']].head(10).to_string())
            
            # Guardar resultado
            resultado.to_csv('predicciones_nuevas.csv', index=False)
            print(f"\n✓ Archivo guardado: predicciones_nuevas.csv")
    
    else:
        # Modo demostración
        print("\n📋 Modo Demostración - Ejemplo de uso:")
        print("\nEjemplo 1 - Valores normales:")
        result = predictor.predecir_simple(
            rpm=1000, accel_b=0.45, vel_b=2.4, env_b=1.6,
            accel_a=0.54, vel_a=1.7, env_a=0.99
        )
        if result:
            print(f"  Estado: {result['estado']}")
            print(f"  Confianza: {result['confianza']:.1f}%")
        
        print("\nEjemplo 2 - Valores anormales:")
        result = predictor.predecir_simple(
            rpm=500, accel_b=1.2, vel_b=5.0, env_b=4.5,
            accel_a=1.5, vel_a=4.2, env_a=3.5
        )
        if result:
            print(f"  Estado: {result['estado']}")
            print(f"  Confianza: {result['confianza']:.1f}%")
        
        print("\nPara usar:")
        print("  python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 \\")
        print("                      --env_b 1.6 --accel_a 0.54 --vel_a 1.7 --env_a 0.99")
        print("\n  python predictor.py --csv nuevo_dataset.csv")


if __name__ == '__main__':
    main()
