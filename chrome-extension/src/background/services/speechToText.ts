import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage } from '@langchain/core/messages';
import { createLogger } from '../log';
import { type ProviderConfig, speechToTextModelStore } from '@extension/storage';
import { t } from '@extension/i18n';

const logger = createLogger('SpeechToText');

export class SpeechToTextService {
  private llm?: ChatGoogleGenerativeAI;
  private serverUrl?: string;
  private type: 'gemini' | 'whisper_cpp';

  private constructor(type: 'gemini' | 'whisper_cpp', llm?: ChatGoogleGenerativeAI, serverUrl?: string) {
    this.type = type;
    this.llm = llm;
    this.serverUrl = serverUrl;
  }

  static async create(providers: Record<string, ProviderConfig>): Promise<SpeechToTextService> {
    try {
      const config = await speechToTextModelStore.getSpeechToTextModel();

      if (!config) {
        throw new Error(t('chat_stt_model_notFound'));
      }

      if (config.type === 'whisper_cpp') {
        if (!config.serverUrl) {
          throw new Error('Whisper.cpp server URL not configured');
        }
        logger.info(`Speech-to-text service created with whisper.cpp at ${config.serverUrl}`);
        return new SpeechToTextService('whisper_cpp', undefined, config.serverUrl);
      }

      // Default to gemini
      if (!config.provider || !config.modelName) {
        throw new Error(t('chat_stt_model_notFound'));
      }

      const provider = providers[config.provider];
      logger.info('Found provider for speech-to-text:', provider ? 'yes' : 'no', provider?.type);

      if (!provider || provider.type !== 'gemini') {
        throw new Error(t('chat_stt_model_notFound'));
      }

      const llm = new ChatGoogleGenerativeAI({
        model: config.modelName,
        apiKey: provider.apiKey,
        temperature: 0.1,
        topP: 0.8,
      });
      logger.info(`Speech-to-text service created with model: ${config.modelName}`);
      return new SpeechToTextService('gemini', llm, undefined);
    } catch (error) {
      logger.error('Failed to create speech-to-text service:', error);
      throw error;
    }
  }

  async transcribeAudio(base64Audio: string, mimeType = 'audio/webm'): Promise<string> {
    try {
      logger.info(`Starting audio transcription using ${this.type}...`);

      if (this.type === 'whisper_cpp' && this.serverUrl) {
        // Convert base64 to raw bytes
        const binaryString = atob(base64Audio);
        const rawBytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          rawBytes[i] = binaryString.charCodeAt(i);
        }

        // Use the mime type from the side panel to determine format
        // Side panel converts to WAV when possible for whisper.cpp compatibility
        const isWav = mimeType === 'audio/wav';
        const fileName = isWav ? 'audio.wav' : 'audio.webm';
        const audioBlob = new Blob([rawBytes.buffer as ArrayBuffer], { type: mimeType });
        logger.info(`Audio received: ${mimeType}, size: ${audioBlob.size} bytes`);

        const baseUrl = this.serverUrl.replace(/\/$/, '');

        // Strategy 1: Try /inference (native whisper.cpp)
        {
          const fd = new FormData();
          fd.append('file', audioBlob, fileName);
          fd.append('temperature', '0.0');
          fd.append('response_format', 'json');

          const resp = await fetch(`${baseUrl}/inference`, { method: 'POST', body: fd });
          if (resp.ok) {
            const data = await resp.json();
            const text = (data.text || '').trim();
            logger.info('Transcription via /inference:', text);
            return text;
          }
          logger.info(`/inference failed: ${resp.status}`);
        }

        // Strategy 2: Try /v1/audio/transcriptions (OpenAI-compatible)
        {
          const fd = new FormData();
          fd.append('file', audioBlob, fileName);
          fd.append('model', 'whisper-1');

          const resp = await fetch(`${baseUrl}/v1/audio/transcriptions`, { method: 'POST', body: fd });
          if (resp.ok) {
            const data = await resp.json();
            const text = (data.text || '').trim();
            logger.info('Transcription via /v1/audio/transcriptions:', text);
            return text;
          }
          let errorDetail = '';
          try {
            errorDetail = await resp.text();
          } catch {
            // ignore
          }
          logger.error(`/v1/audio/transcriptions failed: ${resp.status}`, errorDetail);
        }

        throw new Error('All whisper transcription attempts failed. Check server logs. ' + `Server URL: ${baseUrl}`);
      }

      // Gemini logic
      if (!this.llm) {
        throw new Error('Gemini LLM not initialized');
      }

      const transcriptionMessage = new HumanMessage({
        content: [
          {
            type: 'text',
            text: 'Transcribe this audio. Return only the transcribed text without any additional formatting or explanations.',
          },
          {
            type: 'media',
            data: base64Audio,
            mimeType: 'audio/webm',
          },
        ],
      });

      const transcriptionResponse = await this.llm.invoke([transcriptionMessage]);
      const transcribedText = transcriptionResponse.content.toString().trim();
      logger.info('Audio transcription completed (gemini):', transcribedText);

      return transcribedText;
    } catch (error) {
      logger.error('Failed to transcribe audio:', error);
      throw new Error(`Speech transcription failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}
