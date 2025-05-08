# Valid-KBench
Valid-KBench es un benchmark diseñado para evaluar modelos de lenguaje generativo en tareas de programación funcional. Inspirado en HumanEval, este proyecto amplía la evaluación a múltiples lenguajes de programación y utiliza la métrica pass@k para medir la capacidad de los modelos para generar soluciones correctas en múltiples intentos.
## Características Principales
- Evaluación Multilenguaje: Soporte para Python, C y Bash.

- Métrica pass@k: Implementación oficial para calcular la probabilidad de éxito en k intentos.

- Validación de Código: Ejecución y comparación de salidas para verificar la corrección funcional.

- Análisis Detallado: Generación de archivos CSV con resultados por intento y gráficos comparativos.

- Auditoría Completa: Registro de prompts, respuestas del modelo, código generado y resultados de ejecución.

## Estructura del Proyecto
Valid-KBench/



├── prompts/             # Conjuntos de prompts en formato JSON

└── prompts.json         # Prompt original de pruebas

├── resultados/          # Resultados individuales por modelo en archivo csv

├── resultados_oficiales/ # Resultados agregados en formato .csv y gráfico de promedio global con la métrica pass@3

├── graficas/            # Gráficos generados por el benchmark

   └── codellama7B/      # Gráficas individuales por nombre de modelos

├── graficas_comparativas/ # Gráficos comparativos de todos los modelos

├── auditoria/           # Registros detallados de cada evaluación

├── valid-kBench.py      # Script principal de evaluación

├── requirements.txt     # Dependencias del proyecto

└── README.md            # Documentación del proyecto


## Requisitos
Python 3.8 o superior

Dependencias listadas en requirements.txt

```bash
pip install -r requirements.txt
```
## Uso
1. Preparar los Prompts
Asegúrate de tener un archivo JSON (prompts.json) con los prompts. Cada prompt debe incluir:
```JSON
{
  "Categoria": [
    {
      "description": "Descripción de la tarea",
      "input": "Entrada para el programa",
      "expected_output": "Salida esperada"
    }
  ]
}
```
## Ejecutar la Evaluación
```sh
python valid-kBench.py --alias <nombre_modelo> --model_id <ID-modelo_huggingface>
```
### Ejemplo
```sh
python valid-kBench.py --alias <qwen2.5BASE> --model_id Qwen/Qwen2.5-Coder-7B
```
## Resultados
-CSV Individuales: En resultados/<alias>/, con detalles por intento.
-CSV Agregados: En resultados_oficiales/, con métricas pass@k por prompt.
-Gráficos: En graficas/<alias>/, visualizando el rendimiento por lenguaje y modelo.
-Auditoría: En auditoria/<alias>/, con registros detallados de cada evaluación.

## Métrica pass@k
La métrica pass@k se calcula utilizando la fórmula oficial:

   $\text{pass@k} = 1 - \frac{{\binom{n-c}{k}}}{{\binom{n}{k}}}$

Donde:

- n: Número total de intentos (k).

. c: Número de intentos correctos.

Esta métrica estima la probabilidad de que al menos una de las k respuestas generadas sea correcta.

## Interpretación de Resultados
### Aciero: Indica si un intento individual fue correcto (1.0) o incorrecto (0.0).

![image](https://github.com/user-attachments/assets/b4df51ae-c23a-498b-9311-7357e4c48b15)


### Visualizan el tiempo de inferencia del modelo y ejecución del codigo
![image](https://github.com/user-attachments/assets/5b020c46-c8c6-4c8a-8c33-8263df69a403)

![image](https://github.com/user-attachments/assets/6928803b-a14f-4f26-971b-47b6786a25e4)


### Visualizan el rendimiento promedio por modelo y lenguaje.

![image](https://github.com/user-attachments/assets/c4282b21-d510-479c-90b5-7f8611a2446a)

