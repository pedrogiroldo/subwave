import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

const MODEL_PATH = "src-tauri/models/ggml-base.bin";

function App() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    initialize();

    const unlisten = listen<string>("transcription", (event) => {
      const text = event.payload;
      if (text.trim()) {
        setTranscription((prev) => prev + " " + text);
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  async function initialize() {
    try {
      const loaded = await invoke<boolean>("is_model_loaded");
      setModelLoaded(loaded);
      if (!loaded) {
        setStatus("Carregando modelo...");
        await invoke("load_whisper_model", { path: MODEL_PATH });
        setModelLoaded(true);
      }
      const capturing = await invoke<boolean>("is_capturing");
      setIsCapturing(capturing);
      setStatus(capturing ? "Gravando..." : "Pronto para gravar");
    } catch (err) {
      setStatus(`Erro ao inicializar modelo: ${err}`);
    }
  }

  async function startCapture() {
    if (isCapturing) {
      return;
    }
    setStatus("Iniciando gravação...");
    try {
      await invoke("start_capture");
      setIsCapturing(true);
      setStatus("Gravando...");
    } catch (err) {
      setStatus(`Erro ao iniciar gravação: ${err}`);
    }
  }

  async function stopCapture() {
    if (!isCapturing) {
      return;
    }
    setStatus("Parando gravação...");
    try {
      await invoke("stop_capture");
      setIsCapturing(false);
      setStatus("Pronto para gravar");
    } catch (err) {
      setStatus(`Erro ao parar gravação: ${err}`);
    }
  }

  return (
    <main className="container">
      <h1>SubWave</h1>

      <section className="card">
        <div className="row">
          <button className="start" onClick={startCapture} disabled={!modelLoaded || isCapturing}>
            {isCapturing ? "Gravando..." : "Iniciar gravacao"}
          </button>
          <button className="stop" onClick={stopCapture} disabled={!modelLoaded || !isCapturing}>
            Parar gravacao
          </button>
        </div>
        <p className="status">{status}</p>
      </section>

      <section className="card">
        <h2>Transcricao</h2>
        <div className="transcription-box">
          {transcription || "A transcricao aparece aqui..."}
        </div>
      </section>
    </main>
  );
}

export default App;
