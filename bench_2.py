import os
import json
import time
import re
import subprocess
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse


class ChatModelHF:
    def __init__(self, model_name, device="cuda"):
        self.system_prompt = (
            "Eres una IA que genera código en Python, C y bash según lo que se indique en el prompt "
            "y el código debe imprimir solo el resultado, sin texto adicional."
        )
        self.chat_history = [self.system_prompt]

        print(f"[INFO] Descargando modelo {model_name}...")
        local_dir = snapshot_download(repo_id=model_name)
        print(f"[INFO] Modelo descargado en {local_dir}")

        print(f"[INFO] Cargando modelo {model_name} desde {local_dir} en {device}")
        if "qwen" in model_name.lower():
            os.environ["PYTORCH_USE_SDPA"] = "0"

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.device = device

    def ask(self, prompt, reset_history=True):
        if reset_history:
            self.chat_history = [self.system_prompt]
        full_prompt = "\n".join(self.chat_history + [prompt])

        # Tokenizamos y movemos tensores a GPU/CPU
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        t1 = time.perf_counter()

        # Separamos sólo los tokens generados, descartando los del prompt
        input_len = inputs["input_ids"].shape[-1]
        gen_tokens = output[0][input_len:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        self.chat_history.extend([prompt, response])
        return response, t1 - t0


    @staticmethod
    def extract_code(text, language):
        pattern_md = rf"```{language}(.*?)```"
        match = re.search(pattern_md, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = text.strip().splitlines()
        code_lines = []
        inside_code = False
        for line in lines:
            if language == "python" and line.strip().startswith(
                ("def ", "print(", "for ", "if ", "while ", "#", "import ")
            ):
                inside_code = True
            elif language == "bash" and line.strip().startswith(
                ("echo ", "#!", "for ", "if ", "while ")
            ):
                inside_code = True
            elif language == "c" and line.strip().startswith(
                ("#include", "int main", "printf", "scanf")
            ):
                inside_code = True
            if inside_code:
                code_lines.append(line)
        return "\n".join(code_lines).strip() if code_lines else ""

    @staticmethod
    def cargar_prompts(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class CodeValidatorImproved:
    def __init__(self):
        self.extensions = {"python": ".py", "c": ".c", "bash": ".bash"}

    def validar_stdout(self, code, language, entrada):
        filename = f"codigo_gen{self.extensions[language]}"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            t0 = time.perf_counter()
            if language == "python":
                proc = subprocess.run(
                    ["python", filename],
                    input=str(entrada),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif language == "c":
                subprocess.run(["gcc", filename, "-o", "prog"], check=True, timeout=10)
                proc = subprocess.run(
                    ["./prog"],
                    input=str(entrada),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif language == "bash":
                proc = subprocess.run(
                    ["bash", filename],
                    input=str(entrada),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                return False, "Lenguaje no soportado", None
            t1 = time.perf_counter()
            return proc.returncode == 0, proc.stdout.strip(), t1 - t0
        except Exception as e:
            return False, str(e), None

    def validar_assert_python(self, code, test_code):
        filename = "codigo_assert.py"
        full_code = code + "\n\n" + test_code
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_code)
            proc = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True,
                timeout=10
            )
            return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
        except Exception as e:
            return False, "", str(e)


class Evaluador:
    def __init__(self, model_alias, model_id):
        self.model_alias = model_alias
        self.model = ChatModelHF(model_id)
        self.validator = CodeValidatorImproved()
        self.resultados = []
        self.auditoria_dir = f"auditoria/{model_alias}"
        os.makedirs(self.auditoria_dir, exist_ok=True)
        self.audit_file = os.path.join(self.auditoria_dir, "auditoria.txt")

    def log_auditoria(self, prompt, response, code, lang, esperado, salida, extra=""):
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Lenguaje: {lang}\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Respuesta del modelo:\n{response}\n\n")
            f.write(f"Código extraído:\n{code if code else '[NO DETECTADO]'}\n\n")
            f.write(f"Esperado: {esperado}\n")
            f.write(f"Salida   : {salida}\n")
            if extra:
                f.write(f"Info extra: {extra}\n")

    def evaluar(self, prompts_json):
        prompts = ChatModelHF.cargar_prompts(prompts_json)
        for categoria, tests in tqdm(prompts.items(), desc="📂 Categorías", position=0):
            for test in tqdm(tests, desc=f"📄 Prompts ({categoria})", leave=False, position=1):
                desc = test["description"]
                entrada = test["input"]
                esperado = str(test["expected_output"]).strip()
                func_name = test.get("function_name", "mi_funcion")
                test_code = test.get("test_code", "")

                for lang in ["python", "c", "bash"]:
                    if lang == "python" and test_code:
                        prompt = (
                            f"Escribe una función llamada `{func_name}` en {lang} que {desc} {entrada}. "
                            f"La función debe pasar el siguiente test:\n{test_code}"
                        )
                    else:
                        prompt = (
                            f"Escribe un programa en {lang} que {desc} {entrada}. "
                            f"El programa debe imprimir exactamente:\n{esperado}"
                        )

                    respuesta, t_modelo = self.model.ask(prompt, reset_history=True)
                    codigo = self.model.extract_code(respuesta, lang)

                    if not codigo:
                        self.log_auditoria(prompt, respuesta, None, lang, esperado, "[NO OUTPUT]")
                        self.resultados.append({
                            "modelo": self.model_alias,
                            "categoria": categoria,
                            "lenguaje": lang,
                            "tiempo_modelo": round(t_modelo, 4),
                            "tiempo_ejecucion": 0.0,
                            "acierto": 0
                        })
                        continue

                    if lang == "python" and test_code:
                        correcto, salida, extra = self.validator.validar_assert_python(codigo, test_code)
                        acierto = int(correcto)
                        t_ejec = 0.0
                    else:
                        correcto, salida, t_ejec = self.validator.validar_stdout(codigo, lang, entrada)
                        acierto = int(correcto and salida.strip() == esperado.strip())
                        extra = ""

                    print(f"[{self.model_alias}] [{lang}] {'✅' if acierto else '❌'} Resultado: {salida} / Esperado: {esperado}")
                    self.log_auditoria(prompt, respuesta, codigo, lang, esperado, salida, extra)

                    self.resultados.append({
                        "modelo": self.model_alias,
                        "categoria": categoria,
                        "lenguaje": lang,
                        "tiempo_modelo": round(t_modelo, 4),
                        "tiempo_ejecucion": round(t_ejec or 0.0, 4),
                        "acierto": acierto
                    })

    def exportar_resultados(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.resultados)
        df.to_csv(path, index=False)
        print(f" Resultados guardados en {path}")

    def graficar(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.resultados)
        if df.empty:
            print(" No hay resultados para graficar.")
            return

        modelos = df["modelo"].unique()
        for modelo in modelos:
            df_modelo = df[df["modelo"] == modelo]
            df_agg = df_modelo.groupby("lenguaje").agg({
                "tiempo_modelo": "mean",
                "tiempo_ejecucion": "mean",
                "acierto": ["sum", "count"]
            })
            df_agg.columns = ['t_modelo', 't_ejecucion', 'aciertos', 'intentos']
            df_agg["porcentaje"] = df_agg["aciertos"] / df_agg["intentos"] * 100

            fig, ax = plt.subplots(figsize=(8, 6))
            df_agg["porcentaje"].plot(kind="bar", ax=ax)
            ax.set_title(f"Aciertos por lenguaje - {modelo}")
            ax.set_ylabel("Porcentaje de aciertos")
            ax.set_ylim(0, 100)
            plt.xticks(rotation=0)
            plt.tight_layout()
            path = os.path.join(out_dir, f"{modelo}_aciertos.png")
            plt.savefig(path)
            plt.close()
            print(f"📊 Gráfico guardado en {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluador de modelos de código con Hugging Face")
    parser.add_argument("--alias", required=True, help="Nombre corto del modelo (para carpetas)")
    parser.add_argument("--model_id", required=True, help="ID del modelo en Hugging Face")
    parser.add_argument("--prompts", default="n_p.json", help="Ruta al archivo JSON de prompts")
    args = parser.parse_args()

    evaluador = Evaluador(args.alias, args.model_id)
    evaluador.evaluar(args.prompts)

    out_dir_resultados = f"resultados/{args.alias}"
    out_file = f"{out_dir_resultados}/resultados.csv"
    evaluador.exportar_resultados(path=out_file)

    out_dir_graficas = f"graficas/{args.alias}"
    evaluador.graficar(out_dir=out_dir_graficas)


if __name__ == "__main__":
    main()
