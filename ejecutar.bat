@echo off
REM ============================================================================
REM  DESFIBRADORA - Script de Inicio Rápido (Windows)
REM ============================================================================
REM Descomentar la línea que desees ejecutar

echo.
echo ================================================================================
echo  DESFIBRADORA - Sistema de Predicción de Fallas
echo ================================================================================
echo.
echo Opciones disponibles:
echo.
echo  1. python analisis_prediccion_fallas.py
echo     └─ Ejecutar análisis completo y entrenar modelo
echo.
echo  2. python predictor.py
echo     └─ Hacer predicción demo
echo.
echo  3. python predictor.py --rpm 1000 --accel_b 0.45 --vel_b 2.4 --env_b 1.6 --accel_a 0.54 --vel_a 1.7 --env_a 0.99
echo     └─ Predicción con valores específicos
echo.
echo  4. python monitor_realtime.py
echo     └─ Monitor en tiempo real
echo.
echo ================================================================================
echo.

REM Descomenta la línea siguiente para ejecutar el análisis:
REM python analisis_prediccion_fallas.py

REM Descomenta la línea siguiente para ver predicciones:
REM python predictor.py

REM Descomenta la línea siguiente para el monitor en vivo:
REM python monitor_realtime.py

pause
