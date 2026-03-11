import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

const MODEL_PATH = "src-tauri/models/ggml-base.bin";
const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "pt", label: "Português" },
];

function App() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [translation, setTranslation] = useState("");
  const [status, setStatus] = useState("");
  const [translateReady, setTranslateReady] = useState(false);
  const [translateStatus, setTranslateStatus] = useState("");
  const [srcLang, setSrcLang] = useState("en");
  const [tgtLang, setTgtLang] = useState("pt");
  const translateRequestId = useRef(0);
  const srcLangRef = useRef(srcLang);
  const tgtLangRef = useRef(tgtLang);

  useEffect(() => {
    initialize();

    const unlisten = listen<string>("transcription", (event) => {
      const text = event.payload;
      if (text.trim()) {
        setTranscription(() => text);
        void translateLatest(text);
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    srcLangRef.current = srcLang;
  }, [srcLang]);

  useEffect(() => {
    tgtLangRef.current = tgtLang;
  }, [tgtLang]);

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

      setTranslateStatus("Carregando tradutor...");
      await invoke("init_translate_model");
      setTranslateReady(true);
      setTranslateStatus("Tradutor pronto");
    } catch (err) {
      setStatus(`Erro ao inicializar modelo: ${err}`);
      setTranslateStatus(`Erro ao inicializar tradutor: ${err}`);
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

  async function translateLatest(text: string) {
    if (!translateReady) {
      return;
    }
    const requestId = ++translateRequestId.current;
    setTranslateStatus("Traduzindo...");
    try {
      const currentSrc = srcLangRef.current;
      const currentTgt = tgtLangRef.current;
      if (currentSrc === currentTgt) {
        if (requestId === translateRequestId.current) {
          setTranslation(text);
          setTranslateStatus("Tradutor pronto");
        }
        return;
      }
      const translated = await invoke<string>("translate_text", {
        text,
        srcLang: currentSrc,
        tgtLang: currentTgt,
      });
      if (requestId === translateRequestId.current) {
        setTranslation(translated);
        setTranslateStatus("Traducao pronta");
      }
    } catch (err) {
      if (requestId === translateRequestId.current) {
        setTranslateStatus(`Erro ao traduzir: ${err}`);
      }
    }
  }

  return (
    <main className="container">
      <h1>SubWave</h1>

      <section className="card">
        <div className="row">
          <button
            className="start"
            onClick={startCapture}
            disabled={!modelLoaded || isCapturing}
          >
            {isCapturing ? "Gravando..." : "Iniciar gravacao"}
          </button>
          <button
            className="stop"
            onClick={stopCapture}
            disabled={!modelLoaded || !isCapturing}
          >
            Parar gravacao
          </button>
        </div>
        <p className="status">{status}</p>
      </section>

      <section className="card">
        <h2>Traducao</h2>
        <div className="row select-row">
          <label className="select-group">
            <span>Origem</span>
            <select value={srcLang} onChange={(e) => setSrcLang(e.target.value)}>
              {LANGUAGES.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
          </label>
          <label className="select-group">
            <span>Destino</span>
            <select value={tgtLang} onChange={(e) => setTgtLang(e.target.value)}>
              {LANGUAGES.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="status">{translateStatus}</p>
        <div className="transcription-box">
          {translation || transcription || "A traducao aparece aqui..."}
        </div>
      </section>
    </main>
  );
}

export default App;
