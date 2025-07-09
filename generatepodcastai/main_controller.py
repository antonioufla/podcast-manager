from generatepodcastai.input_agents import TextInputAgent, URLInputAgent, VideoInputAgent
from generatepodcastai.core_agents import DocumentProcessingAgent, QAAgent, SummarizationAgent, IdeationAgent, TopicExtractionAgent
from generatepodcastai.audio_agents import ScriptGenerationAgent, VoiceSynthesisAgent, AudioMixingAgent

class MainController:
    def __init__(self, llm, tts_api_client):
        self.text_input_agent = TextInputAgent()
        self.url_input_agent = URLInputAgent()
        self.video_input_agent = VideoInputAgent()
        self.document_processing_agent = DocumentProcessingAgent()
        self.qa_agent = QAAgent()
        self.summarization_agent = SummarizationAgent()
        self.ideation_agent = IdeationAgent()
        self.topic_extraction_agent = TopicExtractionAgent()
        self.script_generation_agent = ScriptGenerationAgent(llm)
        self.voice_synthesis_agent = VoiceSynthesisAgent(tts_api_client)
        self.audio_mixing_agent = AudioMixingAgent()

    def process_text_files(self, filepaths):
        texts = []
        for path in filepaths:
            text = self.text_input_agent.load_text_file(path)
            texts.append(text)
        return texts

    def process_url(self, url):
        content = self.url_input_agent.fetch_url_content(url)
        if content is None:
            # It's a video URL
            video_path = self.video_input_agent.download_video(url)
            audio_path = self.video_input_agent.extract_audio(video_path)
            transcript = self.video_input_agent.transcribe_audio(audio_path)
            return [transcript]
        else:
            return [content]

    def process_documents(self, texts):
        vectorstore = self.document_processing_agent.process_documents(texts)
        return vectorstore

    def answer_question(self, question, vectorstore):
        return self.qa_agent.answer_question(question, vectorstore)

    def generate_summary(self, text):
        return self.summarization_agent.summarize(text)

    def generate_ideas(self, prompt):
        return self.ideation_agent.generate_ideas(prompt)

    def extract_topics(self, text):
        return self.topic_extraction_agent.extract_topics(text)

    def generate_podcast_script(self, documents, tone="conversational"):
        return self.script_generation_agent.generate_script(documents, tone)

    def synthesize_voices(self, script_text, voice_ids):
        audio_files = []
        for voice_id in voice_ids:
            audio_path = self.voice_synthesis_agent.synthesize_voice(script_text, voice_id)
            audio_files.append(audio_path)
        return audio_files

    def mix_audio_files(self, audio_files, output_path="final_mix.mp3"):
        return self.audio_mixing_agent.mix_audios(audio_files, output_path)
