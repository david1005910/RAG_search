#!/usr/bin/env python3
"""
Medical/Scientific Paper RAG System
- 대화형 질의응답 지원
- 논문 요약 기능 (OpenAI API)
- 다국어 지원 (영어/한국어)
- .env 파일을 통한 API 키 관리
"""
#!pip install langchain langchain-community langchain-text-splitters langchain-openai faiss-cpu

import os
import requests
import time
import re
from pathlib import Path
from typing import List, Dict, Optional

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 논문 검색
import arxiv
import xmltodict
# PDF 처리
from PyPDF2 import PdfReader
import pdfplumber

# LangChain 관련
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# 디렉토리 설정
PAPERS_DIR = "./papers"
VECTORSTORE_DIR = "./vectorstore"
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ==================== 언어 감지 및 번역 ====================
def detect_language(text: str) -> str:
    """텍스트의 언어를 감지 (한국어 또는 영어)"""
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.findall(r'[a-zA-Z가-힣]', text))

    if total_chars == 0:
        return 'en'

    korean_ratio = korean_chars / total_chars
    return 'ko' if korean_ratio > 0.3 else 'en'


def translate_to_english(text: str, openai_api_key: str = None) -> str:
    """한국어 텍스트를 영어로 번역 (검색용)"""
    if not openai_api_key:
        # API 키가 없으면 간단한 의학 용어 매핑 사용
        medical_terms = {
            '당뇨병': 'diabetes mellitus',
            '당뇨': 'diabetes',
            '치료': 'treatment',
            '치료법': 'treatment therapy',
            '암': 'cancer',
            '폐암': 'lung cancer',
            '유방암': 'breast cancer',
            '위암': 'gastric cancer stomach cancer',
            '간암': 'liver cancer hepatocellular carcinoma',
            '대장암': 'colon cancer colorectal cancer',
            '고혈압': 'hypertension',
            '심장병': 'heart disease cardiovascular disease',
            '뇌졸중': 'stroke cerebrovascular accident',
            '치매': 'dementia alzheimer',
            '우울증': 'depression',
            '비만': 'obesity',
            '골다공증': 'osteoporosis',
            '관절염': 'arthritis',
            '천식': 'asthma',
            '알레르기': 'allergy',
            '감염': 'infection',
            '바이러스': 'virus viral',
            '백신': 'vaccine vaccination',
            '항생제': 'antibiotic',
            '면역': 'immunity immune',
            '진단': 'diagnosis diagnostic',
            '예방': 'prevention preventive',
            '증상': 'symptoms',
            '부작용': 'side effects adverse effects',
            '임상시험': 'clinical trial',
            '약물': 'drug medication',
            '수술': 'surgery surgical',
            '방사선': 'radiation radiotherapy',
            '화학요법': 'chemotherapy',
        }

        translated = text
        for ko, en in medical_terms.items():
            if ko in translated:
                translated = translated.replace(ko, en)

        # 남은 한글 제거
        translated = re.sub(r'[가-힣]+', '', translated).strip()
        return translated if translated else text

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical/scientific translator. Translate the following Korean medical query to English. Only output the English translation, nothing else."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ 번역 실패, 기본 매핑 사용: {str(e)[:30]}")
        return translate_to_english(text, None)  # API 없이 재시도


# ==================== 설정 클래스 ====================
class Config:
    """RAG 시스템 설정"""
    def __init__(self):
        self.search_source = 'pubmed'
        self.search_query = ''  # 원본 검색어 (한국어 가능)
        self.search_query_en = ''  # 영어로 번역된 검색어 (실제 검색용)
        self.max_results = 5
        self.embedding_model = 'pubmedbert'  # 기본값을 PubMedBERT로 변경
        self.sparse_method = 'bm25'  # 'bm25' 또는 'splade'
        self.chunk_size = 1000
        self.chunk_overlap = 200
        # .env 파일에서 API 키 로드
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY') or None
        self.pubmed_email = os.getenv('PUBMED_EMAIL') or None
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or None
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') or None
        self.language = 'en'  # 감지된 언어

    def interactive_setup(self):
        """대화형 설정"""
        print("\n" + "=" * 60)
        print("⚙️  RAG 시스템 설정")
        print("=" * 60)

        # 검색 소스 선택
        print("\n📖 검색 소스 선택:")
        print("   1. PubMed (의학/생물학)")
        print("   2. arXiv (CS/물리/수학)")
        print("   3. 둘 다")
        choice = input("선택 [1]: ").strip() or "1"
        self.search_source = {'1': 'pubmed', '2': 'arxiv', '3': 'both'}.get(choice, 'pubmed')

        # 검색어 입력
        self.search_query = input("\n🔍 검색어 입력: ").strip()
        if not self.search_query:
            self.search_query = "COVID-19 vaccine efficacy"
            print(f"   기본값 사용: {self.search_query}")

        # 언어 감지
        self.language = detect_language(self.search_query)
        lang_name = "한국어" if self.language == 'ko' else "English"
        print(f"   🌐 감지된 언어: {lang_name}")

        # 최대 결과 수
        max_res = input(f"\n📄 최대 논문 수 [{self.max_results}]: ").strip()
        if max_res.isdigit():
            self.max_results = int(max_res)

        # 임베딩 모델 선택
        print("\n🧠 Dense 임베딩 모델 선택:")
        print("   [PubMed/의학 특화 - 로컬 실행]")
        print("   1. PubMedBERT Full (PubMed 전문 학습) [권장]")
        print("   2. PubMedBERT Abstract (PubMed 초록 특화)")
        print("   3. BioBERT (의학/생물학)")
        print("   4. BioLinkBERT (의학 문헌 링크 학습)")
        print("   5. SciBERT (과학 논문 전반)")
        print("   [일반 모델]")
        print("   6. BERT-base (일반)")
        print("   [OpenAI API - 빠르고 정확]")
        print("   7. OpenAI Small (빠름, 저렴)")
        print("   8. OpenAI Large (고성능)")
        model_choice = input("선택 [1 - PubMedBERT]: ").strip() or "1"
        self.embedding_model = {
            '1': 'pubmedbert',
            '2': 'pubmedbert-abs',
            '3': 'biobert',
            '4': 'biolinkbert',
            '5': 'scibert',
            '6': 'bert-base',
            '7': 'openai-small',
            '8': 'openai-large'
        }.get(model_choice, 'pubmedbert')

        # Sparse 검색 방식 선택
        print("\n🔍 Sparse 검색 방식 선택:")
        print("   1. BM25 (전통적 키워드 매칭) [기본값]")
        print("   2. SPLADE (신경망 기반 확장 검색)")
        sparse_choice = input("선택 [1 - BM25]: ").strip() or "1"
        self.sparse_method = {
            '1': 'bm25',
            '2': 'splade'
        }.get(sparse_choice, 'bm25')

        # PubMed API 설정
        if self.search_source in ['pubmed', 'both']:
            print("\n🔑 PubMed API 설정 (선택사항 - 속도 향상):")
            if self.pubmed_api_key:
                print(f"   ✅ .env에서 로드됨 (Key: {self.pubmed_api_key[:8]}...)")
            else:
                print("   API 키가 없으면 Enter를 누르세요.")
                print("   발급: https://www.ncbi.nlm.nih.gov/account/settings/")
                self.pubmed_api_key = input("   API Key: ").strip() or None
                if self.pubmed_api_key:
                    self.pubmed_email = input("   Email: ").strip() or None

        # OpenAI API 설정 (논문 요약 및 번역용)
        print("\n🤖 OpenAI API 설정 (논문 요약 및 한→영 번역용):")
        if self.openai_api_key:
            print(f"   ✅ .env에서 로드됨 (Key: {self.openai_api_key[:12]}...)")
        else:
            print("   API 키가 없으면 Enter를 누르세요.")
            print("   발급: https://platform.openai.com/api-keys")
            self.openai_api_key = input("   OpenAI API Key: ").strip() or None

        # 한국어 검색어인 경우 영어로 번역
        if self.language == 'ko':
            print("\n🔄 한국어 검색어를 영어로 번역 중...")
            self.search_query_en = translate_to_english(self.search_query, self.openai_api_key)
            print(f"   🇰🇷 원본: {self.search_query}")
            print(f"   🇺🇸 번역: {self.search_query_en}")
        else:
            self.search_query_en = self.search_query

        print("\n" + "-" * 60)
        print("✅ 설정 완료!")
        print(f"   📖 소스: {self.search_source}")
        print(f"   🔍 검색어: {self.search_query}")
        if self.language == 'ko':
            print(f"   🔍 검색어(영문): {self.search_query_en}")
        print(f"   🌐 언어: {lang_name} (응답도 {lang_name}로)")
        print(f"   📄 최대 논문: {self.max_results}")
        print(f"   🧠 Dense 모델: {self.embedding_model}")
        print(f"   🔤 Sparse 방식: {self.sparse_method.upper()}")
        if self.pubmed_api_key:
            print(f"   🔑 PubMed API: 설정됨")
        if self.openai_api_key:
            print(f"   🤖 OpenAI API: 설정됨 (요약/번역 활성화)")
        print("-" * 60)

        return self


# ==================== 논문 요약 클래스 ====================
class PaperSummarizer:
    """OpenAI API를 사용한 논문 요약"""

    def __init__(self, api_key: str = None, language: str = 'en'):
        self.api_key = api_key
        self.language = language

    def summarize(self, papers: List[Dict], documents: List[Dict]) -> List[Dict]:
        """논문들을 요약"""
        if not self.api_key:
            print("\n⚠️ OpenAI API 키가 없어 요약을 건너뜁니다.")
            return papers

        print("\n" + "=" * 60)
        print("📝 논문 요약 중... (OpenAI API 사용)")
        print("=" * 60)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("⚠️ OpenAI 패키지가 설치되지 않았습니다. pip install openai")
            return papers
        except Exception as e:
            print(f"⚠️ OpenAI 초기화 실패: {str(e)[:50]}")
            return papers

        # 언어별 프롬프트
        if self.language == 'ko':
            system_prompt = """당신은 의학/과학 논문을 요약하는 전문가입니다.
논문의 제목, 저자, 초록, 본문 내용을 바탕으로 핵심 내용을 한국어로 간결하게 요약해주세요.
요약은 다음 형식을 따르세요:
- 연구 목적
- 주요 방법
- 핵심 결과
- 결론 및 의의"""
        else:
            system_prompt = """You are an expert at summarizing medical/scientific papers.
Based on the title, authors, abstract, and content, provide a concise summary.
Follow this format:
- Research Objective
- Key Methods
- Main Results
- Conclusion & Significance"""

        summarized_papers = []

        for i, paper in enumerate(papers):
            print(f"\n   [{i+1}/{len(papers)}] {paper['title'][:50]}...")

            # 해당 논문의 본문 찾기
            paper_content = ""
            for doc in documents:
                if paper['id'] in doc['source']:
                    paper_content = doc['text'][:3000]  # 토큰 제한
                    break

            # 요약할 내용 구성
            content_to_summarize = f"""
Title: {paper['title']}
Authors: {', '.join(paper['authors'][:5])}
Published: {paper['published']}
Source: {paper['source']}

Abstract:
{paper['abstract']}

Content:
{paper_content}
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content_to_summarize}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )

                summary = response.choices[0].message.content
                paper['summary'] = summary
                print(f"      ✅ 요약 완료")

            except Exception as e:
                print(f"      ⚠️ 요약 실패: {str(e)[:50]}")
                paper['summary'] = paper['abstract'][:500] + "..."

            summarized_papers.append(paper)
            time.sleep(0.5)  # API 속도 제한

        # 요약 결과 출력
        print("\n" + "=" * 60)
        print("📋 논문 요약 결과")
        print("=" * 60)

        for i, paper in enumerate(summarized_papers):
            print(f"\n[{i+1}] {paper['title'][:60]}...")
            print("-" * 40)
            print(paper.get('summary', 'No summary available'))
            print("-" * 40)

        input("\n⏎ Enter를 눌러 다음 단계로 진행...")

        return summarized_papers


# ==================== PaperSearcher 클래스 ====================
class PaperSearcher:
    def __init__(self, api_key: str = None, email: str = None):
        self.papers = []
        self.api_key = api_key
        self.email = email

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        print(f"\n🔍 arXiv에서 '{query}' 검색 중...")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in search.results():
            paper = {
                'source': 'arXiv',
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published.strftime('%Y-%m-%d'),
                'id': result.entry_id.split('/')[-1]
            }
            papers.append(paper)
            print(f"   📄 {paper['title'][:60]}...")

        print(f"   ✅ arXiv: {len(papers)}개 논문 발견")
        return papers

    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        print(f"\n🔍 PubMed에서 '{query}' 검색 중...")

        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }

        if self.api_key:
            search_params['api_key'] = self.api_key
            print("   🔑 API 키 사용 중...")
        if self.email:
            search_params['email'] = self.email

        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                print("   ⚠️ PubMed: 검색 결과 없음")
                return []

            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            if self.api_key:
                fetch_params['api_key'] = self.api_key

            if not self.api_key:
                time.sleep(0.35)

            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            response.raise_for_status()

            data = xmltodict.parse(response.content)
            articles = data.get('PubmedArticleSet', {}).get('PubmedArticle', [])

            if isinstance(articles, dict):
                articles = [articles]

            papers = []
            for article in articles:
                try:
                    medline = article.get('MedlineCitation', {})
                    article_data = medline.get('Article', {})

                    title = article_data.get('ArticleTitle', 'No Title')
                    if isinstance(title, dict):
                        title = title.get('#text', 'No Title')

                    abstract_data = article_data.get('Abstract', {}).get('AbstractText', '')
                    if isinstance(abstract_data, list):
                        abstract = ' '.join([a.get('#text', str(a)) if isinstance(a, dict) else str(a) for a in abstract_data])
                    elif isinstance(abstract_data, dict):
                        abstract = abstract_data.get('#text', str(abstract_data))
                    else:
                        abstract = str(abstract_data) if abstract_data else 'No abstract available'

                    author_list = article_data.get('AuthorList', {}).get('Author', [])
                    if isinstance(author_list, dict):
                        author_list = [author_list]
                    authors = []
                    for author in author_list[:5]:
                        if isinstance(author, dict):
                            last = author.get('LastName', '')
                            first = author.get('ForeName', '')
                            if last:
                                authors.append(f"{last} {first}".strip())

                    pmid = medline.get('PMID', {}).get('#text', 'Unknown')
                    pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                    year = pub_date.get('Year', 'Unknown')

                    paper = {
                        'source': 'PubMed',
                        'title': title,
                        'authors': authors if authors else ['Unknown'],
                        'abstract': abstract,
                        'pdf_url': None,
                        'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'pmc_url': None,
                        'published': str(year),
                        'id': f"PMID_{pmid}"
                    }
                    papers.append(paper)
                    print(f"   📄 {paper['title'][:60]}...")

                except Exception as e:
                    continue

            print(f"   ✅ PubMed: {len(papers)}개 논문 발견")
            return papers

        except Exception as e:
            print(f"   ❌ PubMed 검색 오류: {str(e)}")
            return []

    def search(self, query: str, source: str = 'both', max_results: int = 5) -> List[Dict]:
        papers = []

        if source in ['arxiv', 'both']:
            papers.extend(self.search_arxiv(query, max_results))

        if source in ['pubmed', 'both']:
            papers.extend(self.search_pubmed(query, max_results))

        self.papers = papers
        return papers


# ==================== PDFDownloader 클래스 ====================
class PDFDownloader:
    def __init__(self, save_dir: str = PAPERS_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(self, paper: Dict) -> Optional[str]:
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
        filename = f"{paper['id']}_{safe_title}.pdf"
        filepath = os.path.join(self.save_dir, filename)
        txt_filepath = filepath.replace('.pdf', '.txt')

        if os.path.exists(txt_filepath):
            print(f"   ⏭️ 이미 존재: {os.path.basename(txt_filepath)[:40]}...")
            return txt_filepath
        if os.path.exists(filepath):
            print(f"   ⏭️ 이미 존재: {filename[:40]}...")
            return filepath

        pdf_url = paper.get('pdf_url') or paper.get('pmc_url')

        if not pdf_url:
            if paper['source'] == 'PubMed':
                return self._save_abstract_as_text(paper, filename)
            print(f"   ⚠️ PDF URL 없음: {paper['title'][:40]}...")
            return None

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
            response = requests.get(pdf_url, headers=headers, timeout=60)
            response.raise_for_status()

            if response.content[:5] == b'<html' or response.content[:5] == b'<!DOC':
                print(f"   ⚠️ PDF 접근 불가, 초록 저장: {paper['title'][:30]}...")
                return self._save_abstract_as_text(paper, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"   ✅ 다운로드: {filename[:40]}...")
            return filepath

        except Exception as e:
            print(f"   ⚠️ PDF 다운로드 실패, 초록 저장: {paper['title'][:30]}...")
            return self._save_abstract_as_text(paper, filename)

    def _save_abstract_as_text(self, paper: Dict, filename: str) -> Optional[str]:
        txt_filename = filename.replace('.pdf', '.txt')
        filepath = os.path.join(self.save_dir, txt_filename)

        content = f"""Title: {paper['title']}

Authors: {', '.join(paper['authors'])}

Source: {paper['source']}

Published: {paper['published']}

Abstract:
{paper['abstract']}

URL: {paper.get('pubmed_url', paper.get('pdf_url', 'N/A'))}
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"   📝 초록 저장: {txt_filename[:40]}...")
        return filepath

    def download_all(self, papers: List[Dict]) -> List[str]:
        print("\n📥 논문 다운로드 시작...\n")

        downloaded = []
        for paper in papers:
            filepath = self.download(paper)
            if filepath:
                downloaded.append(filepath)
            time.sleep(0.3)

        print(f"\n📁 총 {len(downloaded)}개 파일 다운로드 완료!")
        return downloaded


# ==================== TextExtractor 클래스 ====================
class TextExtractor:
    @staticmethod
    def extract(filepath: str) -> str:
        if filepath.endswith('.txt'):
            return TextExtractor._extract_from_txt(filepath)
        elif filepath.endswith('.pdf'):
            return TextExtractor._extract_from_pdf(filepath)
        return ""

    @staticmethod
    def _extract_from_txt(filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""

    @staticmethod
    def _extract_from_pdf(filepath: str) -> str:
        text = ""
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if len(text) < 100:
                text = ""
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        except Exception as e:
            print(f"      ⚠️ 추출 오류: {str(e)[:50]}")

        return text.strip()

    @staticmethod
    def extract_all(filepaths: List[str]) -> List[Dict]:
        print("\n📄 텍스트 추출 중...\n")

        documents = []
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            print(f"   📖 {filename[:50]}...")

            text = TextExtractor.extract(filepath)
            if text:
                documents.append({
                    'text': text,
                    'source': filename,
                    'filepath': filepath
                })
                print(f"      ✅ {len(text):,} 글자 추출")
            else:
                print(f"      ⚠️ 텍스트 없음")

        print(f"\n📊 총 {len(documents)}개 문서에서 텍스트 추출 완료!")
        return documents


# ==================== 커스텀 임베딩 클래스 (sentence-transformers 없이) ====================
from langchain_core.embeddings import Embeddings

os.environ["SAFETENSORS_FAST_GPU"] = "1"


class HuggingFaceEmbeddings(Embeddings):
    """HuggingFace Transformers를 직접 사용하는 임베딩 클래스 (sentence-transformers 불필요)"""

    def __init__(self, model_name: str, device: str = 'cpu'):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = device
        self.model_name = model_name

        print(f"   📥 모델 로드 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - attention mask를 고려한 평균"""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩으로 변환"""
        import torch
        import torch.nn.functional as F

        # 토큰화
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # 임베딩 생성
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트 임베딩"""
        # 배치 처리
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self._encode(batch)
            all_embeddings.extend(embeddings)
            if len(texts) > batch_size:
                print(f"   진행: {min(i+batch_size, len(texts))}/{len(texts)}", end='\r')

        if len(texts) > batch_size:
            print()
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        return self._encode([text])[0]


class OpenAIEmbeddings(Embeddings):
    """OpenAI API를 사용한 임베딩 클래스"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536 if "small" in model else 3072

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트 임베딩"""
        all_embeddings = []
        batch_size = 100  # OpenAI 배치 제한

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            if len(texts) > batch_size:
                print(f"   진행: {min(i+batch_size, len(texts))}/{len(texts)}", end='\r')

        if len(texts) > batch_size:
            print()
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


# ==================== EmbeddingModelFactory 클래스 ====================
class EmbeddingModelFactory:
    """임베딩 모델 팩토리 - HuggingFace 또는 OpenAI 선택 가능"""

    MODELS = {
        # PubMed/의학 특화 모델
        'pubmedbert': {
            'name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            'description': 'PubMed 논문 특화 (PubMedBERT Full)',
            'dimension': 768,
            'type': 'huggingface'
        },
        'pubmedbert-abs': {
            'name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            'description': 'PubMed 초록 특화 (PubMedBERT Abstract)',
            'dimension': 768,
            'type': 'huggingface'
        },
        'biobert': {
            'name': 'dmis-lab/biobert-base-cased-v1.2',
            'description': '의학/생물학 특화 (BioBERT v1.2)',
            'dimension': 768,
            'type': 'huggingface'
        },
        'scibert': {
            'name': 'allenai/scibert_scivocab_uncased',
            'description': '과학 논문 특화 (SciBERT)',
            'dimension': 768,
            'type': 'huggingface'
        },
        'biolinkbert': {
            'name': 'michiyasunaga/BioLinkBERT-base',
            'description': '의학 문헌 링크 학습 (BioLinkBERT)',
            'dimension': 768,
            'type': 'huggingface'
        },
        # 일반 모델
        'bert-base': {
            'name': 'bert-base-uncased',
            'description': 'BERT 기본 모델',
            'dimension': 768,
            'type': 'huggingface'
        },
        # OpenAI API 모델
        'openai-small': {
            'name': 'text-embedding-3-small',
            'description': 'OpenAI 임베딩 (빠름, 저렴)',
            'dimension': 1536,
            'type': 'openai'
        },
        'openai-large': {
            'name': 'text-embedding-3-large',
            'description': 'OpenAI 임베딩 (고성능)',
            'dimension': 3072,
            'type': 'openai'
        }
    }

    @classmethod
    def create(cls, model_type: str = 'biobert', device: str = 'cpu', openai_api_key: str = None):
        if model_type not in cls.MODELS:
            print(f"⚠️ 알 수 없는 모델: {model_type}. 'biobert' 사용")
            model_type = 'biobert'

        model_info = cls.MODELS[model_type]

        print(f"\n🧠 임베딩 모델 로딩 중...")
        print(f"   모델: {model_type}")
        print(f"   설명: {model_info['description']}")
        print(f"   차원: {model_info['dimension']}")

        # OpenAI 모델인 경우
        if model_info['type'] == 'openai':
            if not openai_api_key:
                print("⚠️ OpenAI API 키가 없습니다. biobert로 대체합니다.")
                return cls.create('biobert', device, None)

            try:
                embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=model_info['name'])
                print(f"✅ {model_type} 모델 로드 완료!")
                return embeddings
            except Exception as e:
                print(f"⚠️ OpenAI 임베딩 실패: {str(e)[:50]}")
                print("   biobert로 대체합니다...")
                return cls.create('biobert', device, None)

        # HuggingFace 모델인 경우
        try:
            embeddings = HuggingFaceEmbeddings(model_name=model_info['name'], device=device)
            print(f"✅ {model_type} 모델 로드 완료!")
            return embeddings

        except Exception as e:
            print(f"⚠️ {model_type} 로드 실패: {str(e)[:100]}")
            print("   bert-base 모델로 대체합니다...")

            try:
                return HuggingFaceEmbeddings(
                    model_name='bert-base-uncased',
                    device=device
                )
            except Exception as e2:
                print(f"⚠️ bert-base도 실패: {str(e2)[:50]}")
                print("   OpenAI 임베딩을 사용합니다...")
                if openai_api_key:
                    return OpenAIEmbeddings(api_key=openai_api_key)
                raise RuntimeError("사용 가능한 임베딩 모델이 없습니다.")


# ==================== RAGSystem 클래스 ====================
class RAGSystem:
    def __init__(self, embeddings, chunk_size: int = 1000, chunk_overlap: int = 200, language: str = 'en'):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.language = language
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def build_vectorstore(self, documents: List[Dict]) -> FAISS:
        print("\n✂️ 텍스트 청킹 중...")

        all_chunks = []
        all_metadata = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc['text'])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': doc['source'],
                    'chunk_id': i
                })
            print(f"   📄 {doc['source'][:40]}...: {len(chunks)} 청크")

        print(f"\n📊 총 청크 수: {len(all_chunks)}개")

        if not all_chunks:
            raise ValueError("청킹된 텍스트가 없습니다!")

        print("\n💾 벡터 DB 생성 중...")

        self.vectorstore = FAISS.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadata
        )

        print("✅ 벡터 DB 생성 완료!")
        return self.vectorstore

    def save_vectorstore(self, path: str = VECTORSTORE_DIR):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"💾 벡터 DB 저장 완료: {path}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            print("❌ 벡터 스토어가 없습니다.")
            return []

        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs_with_scores:
            results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'score': float(score)
            })

        return results

    def answer(self, question: str, k: int = 3) -> Dict:
        # 질문 언어 감지
        q_language = detect_language(question)

        results = self.search(question, k=k)

        return {
            'question': question,
            'contexts': results,
            'sources': list(set([r['source'] for r in results])),
            'language': q_language
        }

    def get_all_chunks(self) -> List[Dict]:
        """저장된 모든 청크와 메타데이터 반환"""
        if not self.vectorstore:
            return []

        # FAISS에서 모든 문서 가져오기
        docstore = self.vectorstore.docstore
        index_to_id = self.vectorstore.index_to_docstore_id

        chunks = []
        for idx, doc_id in index_to_id.items():
            doc = docstore.search(doc_id)
            if doc:
                chunks.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', idx)
                })
        return chunks


# ==================== SPLADE Encoder ====================
class SPLADEEncoder:
    """SPLADE (Sparse Lexical and Expansion) 인코더"""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str = 'cpu'):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.device = device
        self.model_name = model_name

        print(f"   📥 SPLADE 모델 로드 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.torch = torch

    def encode(self, texts: List[str], max_length: int = 256) -> List[Dict[str, float]]:
        """텍스트를 SPLADE 스파스 벡터로 인코딩"""
        sparse_vectors = []

        for text in texts:
            # 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # SPLADE 인코딩
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                # SPLADE: log(1 + ReLU(logits)) * attention_mask
                logits = outputs.logits
                relu_log = self.torch.log1p(self.torch.relu(logits))
                # Max pooling over sequence
                weighted = relu_log * inputs['attention_mask'].unsqueeze(-1)
                sparse_vec = self.torch.max(weighted, dim=1).values.squeeze()

            # 0이 아닌 값만 딕셔너리로 저장
            sparse_dict = {}
            indices = self.torch.nonzero(sparse_vec).squeeze(-1)
            for idx in indices:
                idx = idx.item()
                token = self.tokenizer.decode([idx])
                weight = sparse_vec[idx].item()
                if weight > 0.1:  # 임계값 이상만 저장
                    sparse_dict[token] = weight

            sparse_vectors.append(sparse_dict)

        return sparse_vectors

    def compute_similarity(self, query_vec: Dict[str, float], doc_vec: Dict[str, float]) -> float:
        """쿼리와 문서 스파스 벡터의 유사도 계산 (내적)"""
        score = 0.0
        for token, weight in query_vec.items():
            if token in doc_vec:
                score += weight * doc_vec[token]
        return score


# ==================== Hybrid Search 시스템 ====================
class HybridSearchSystem:
    """Sparse (BM25/SPLADE) + Dense (Semantic) + Hybrid 검색 시스템"""

    def __init__(self, rag_system: RAGSystem, sparse_method: str = 'bm25'):
        """
        Args:
            rag_system: RAG 시스템
            sparse_method: 'bm25' 또는 'splade'
        """
        from rank_bm25 import BM25Okapi
        import numpy as np

        self.rag = rag_system
        self.chunks = rag_system.get_all_chunks()
        self.np = np
        self.sparse_method = sparse_method.lower()

        # 스테머 초기화 (BM25용)
        self._init_stemmer()

        # Sparse 인덱스 구축
        if self.sparse_method == 'splade':
            self._init_splade()
        else:
            self._init_bm25()

    def _init_bm25(self):
        """BM25 인덱스 구축"""
        from rank_bm25 import BM25Okapi

        print("\n🔍 BM25 인덱스 구축 중 (스테밍 적용)...")
        tokenized_corpus = [self._tokenize(chunk['content']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.splade = None
        self.splade_vectors = None
        print(f"✅ BM25 인덱스 구축 완료! ({len(self.chunks)} 청크)")

    def _init_splade(self):
        """SPLADE 인덱스 구축"""
        print("\n🔍 SPLADE 인덱스 구축 중...")

        try:
            self.splade = SPLADEEncoder()
            self.bm25 = None

            # 모든 문서 인코딩
            print(f"   📄 {len(self.chunks)}개 문서 인코딩 중...")
            doc_texts = [chunk['content'] for chunk in self.chunks]

            # 배치 처리
            batch_size = 8
            self.splade_vectors = []
            for i in range(0, len(doc_texts), batch_size):
                batch = doc_texts[i:i+batch_size]
                vectors = self.splade.encode(batch)
                self.splade_vectors.extend(vectors)
                print(f"   진행: {min(i+batch_size, len(doc_texts))}/{len(doc_texts)}", end='\r')

            print(f"\n✅ SPLADE 인덱스 구축 완료! ({len(self.chunks)} 청크)")

        except Exception as e:
            print(f"⚠️ SPLADE 로드 실패: {str(e)[:50]}")
            print("   BM25로 대체합니다...")
            self.sparse_method = 'bm25'
            self._init_bm25()

    def _init_stemmer(self):
        """스테머 초기화 - Porter Stemmer 사용"""
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
            self.use_stemming = True
        except ImportError:
            self.stemmer = None
            self.use_stemming = False

    def _tokenize(self, text: str) -> List[str]:
        """토크나이저 + 스테밍"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)

        # 스테밍 적용 (diabete -> diabet, diabetes -> diabet)
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def sparse_search(self, query: str, k: int = 5) -> List[Dict]:
        """Sparse 검색 (BM25 또는 SPLADE)"""
        if self.sparse_method == 'splade' and self.splade is not None:
            return self._splade_search(query, k)
        else:
            return self._bm25_search(query, k)

    def _bm25_search(self, query: str, k: int = 5) -> List[Dict]:
        """BM25 기반 검색"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 인덱스
        top_indices = self.np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunks[idx]['content'],
                'source': self.chunks[idx]['source'],
                'score': float(scores[idx]),
                'method': 'sparse (BM25)'
            })
        return results

    def _splade_search(self, query: str, k: int = 5) -> List[Dict]:
        """SPLADE 기반 검색"""
        # 쿼리 인코딩
        query_vec = self.splade.encode([query])[0]

        # 모든 문서와 유사도 계산
        scores = []
        for doc_vec in self.splade_vectors:
            score = self.splade.compute_similarity(query_vec, doc_vec)
            scores.append(score)

        scores = self.np.array(scores)

        # 상위 k개 인덱스
        top_indices = self.np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunks[idx]['content'],
                'source': self.chunks[idx]['source'],
                'score': float(scores[idx]),
                'method': 'sparse (SPLADE)'
            })
        return results

    def dense_search(self, query: str, k: int = 5) -> List[Dict]:
        """FAISS 기반 Dense 검색 (의미적 유사도)"""
        results = self.rag.search(query, k=k)
        for r in results:
            r['method'] = 'dense'
        return results

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5, rrf_k: int = 10) -> List[Dict]:
        """
        Hybrid 검색 (Sparse + Dense 결합) - RRF (Reciprocal Rank Fusion) 사용
        alpha: 0.0 = 순수 Sparse, 1.0 = 순수 Dense
        rrf_k: RRF 상수 (기본값 10 - 점수 범위 향상)
        """
        # 충분한 후보 검색
        num_candidates = max(k * 3, 20)
        sparse_results = self.sparse_search(query, k=num_candidates)
        dense_results = self.dense_search(query, k=num_candidates)

        # BM25 점수 정규화를 위한 최대/최소값 계산
        bm25_scores = [r['score'] for r in sparse_results]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        # Dense 점수 정규화를 위한 최대/최소값 계산 (L2 거리 - 낮을수록 좋음)
        dense_scores = [r['score'] for r in dense_results]
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_min = min(dense_scores) if dense_scores else 0.0
        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0

        # 문서 통합을 위한 딕셔너리
        doc_data = {}

        # Sparse 결과 처리 - 순위 기반 점수 + BM25 정규화
        for rank, r in enumerate(sparse_results):
            key = r['content'][:100]
            sparse_rrf = 1.0 / (rrf_k + rank + 1)  # RRF 점수
            # BM25 점수를 0-1로 정규화
            bm25_norm = (r['score'] - bm25_min) / bm25_range if bm25_range > 0 else 1.0

            if key not in doc_data:
                doc_data[key] = {
                    'content': r['content'],
                    'source': r['source'],
                    'sparse_rank': rank + 1,
                    'sparse_score': r['score'],  # 원본 BM25 점수 (0~30+ 범위)
                    'sparse_score_norm': bm25_norm,  # 정규화된 BM25 (0-1)
                    'sparse_rrf': sparse_rrf,
                    'dense_rank': 0,
                    'dense_score': 0,
                    'dense_score_norm': 0,
                    'dense_rrf': 0
                }
            else:
                doc_data[key]['sparse_rank'] = rank + 1
                doc_data[key]['sparse_score'] = r['score']
                doc_data[key]['sparse_score_norm'] = bm25_norm
                doc_data[key]['sparse_rrf'] = sparse_rrf

        # Dense 결과 처리 - 순위 기반 점수 + 거리 정규화
        for rank, r in enumerate(dense_results):
            key = r['content'][:100]
            dense_rrf = 1.0 / (rrf_k + rank + 1)  # RRF 점수
            # L2 거리를 유사도로 변환 (1 - 정규화된 거리)
            dense_norm = 1.0 - ((r['score'] - dense_min) / dense_range) if dense_range > 0 else 1.0

            if key not in doc_data:
                doc_data[key] = {
                    'content': r['content'],
                    'source': r['source'],
                    'sparse_rank': 0,
                    'sparse_score': 0,
                    'sparse_score_norm': 0,
                    'sparse_rrf': 0,
                    'dense_rank': rank + 1,
                    'dense_score': r['score'],  # 원본 L2 거리
                    'dense_score_norm': dense_norm,  # 정규화된 유사도 (0-1, 높을수록 좋음)
                    'dense_rrf': dense_rrf
                }
            else:
                doc_data[key]['dense_rank'] = rank + 1
                doc_data[key]['dense_score'] = r['score']
                doc_data[key]['dense_score_norm'] = dense_norm
                doc_data[key]['dense_rrf'] = dense_rrf

        # Hybrid 스코어 계산 (정규화된 점수 기반)
        results = []
        for key, data in doc_data.items():
            # 방법 1: RRF 기반 하이브리드
            hybrid_rrf = (1 - alpha) * data['sparse_rrf'] + alpha * data['dense_rrf']

            # 방법 2: 정규화된 점수 기반 하이브리드 (더 직관적)
            hybrid_norm = (1 - alpha) * data['sparse_score_norm'] + alpha * data['dense_score_norm']

            results.append({
                'content': data['content'],
                'source': data['source'],
                # 원본 점수들
                'sparse_score': data['sparse_score'],      # 원본 BM25 (0~30+)
                'dense_score': data['dense_score'],        # 원본 L2 거리
                # 정규화된 점수들 (0-1)
                'sparse_score_norm': data['sparse_score_norm'],  # BM25 정규화
                'dense_score_norm': data['dense_score_norm'],    # 유사도 정규화
                # 순위 정보
                'sparse_rank': data['sparse_rank'],
                'dense_rank': data['dense_rank'],
                # RRF 점수
                'sparse_rrf': data['sparse_rrf'],
                'dense_rrf': data['dense_rrf'],
                # 하이브리드 점수
                'hybrid_score': hybrid_norm,        # 정규화 기반 (0-1)
                'hybrid_score_rrf': hybrid_rrf,     # RRF 기반
                'method': 'hybrid'
            })

        # Hybrid 스코어로 정렬
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return results[:k]

    def compare_all(self, query: str, k: int = 5, alpha: float = 0.5) -> Dict:
        """세 가지 검색 방법 비교"""
        sparse = self.sparse_search(query, k)
        dense = self.dense_search(query, k)
        hybrid = self.hybrid_search(query, k, alpha)

        return {
            'query': query,
            'sparse': sparse,
            'dense': dense,
            'hybrid': hybrid
        }

    def visualize_comparison(self, query: str, k: int = 5, alpha: float = 0.5,
                            save_path: str = None) -> None:
        """검색 결과 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 비대화형 백엔드

        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        results = self.compare_all(query, k, alpha)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Hybrid Search Comparison\nQuery: "{query[:50]}..."', fontsize=14, fontweight='bold')

        # 1. Sparse (BM25) 점수
        ax1 = axes[0, 0]
        sparse_labels = [f"Doc {i+1}" for i in range(len(results['sparse']))]
        sparse_scores = [r['score'] for r in results['sparse']]
        bars1 = ax1.barh(sparse_labels, sparse_scores, color='#3498db', alpha=0.8)
        ax1.set_xlabel('BM25 Score')
        ax1.set_title('Sparse Search (BM25)', fontweight='bold')
        ax1.invert_yaxis()
        for bar, score in zip(bars1, sparse_scores):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=9)

        # 2. Dense (Semantic) 점수
        ax2 = axes[0, 1]
        dense_labels = [f"Doc {i+1}" for i in range(len(results['dense']))]
        dense_scores = [r['score'] for r in results['dense']]
        bars2 = ax2.barh(dense_labels, dense_scores, color='#e74c3c', alpha=0.8)
        ax2.set_xlabel('L2 Distance (lower = better)')
        ax2.set_title('Dense Search (Semantic)', fontweight='bold')
        ax2.invert_yaxis()
        for bar, score in zip(bars2, dense_scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)

        # 3. Hybrid 정규화 점수 비교
        ax3 = axes[1, 0]
        hybrid_labels = [f"Doc {i+1}" for i in range(len(results['hybrid']))]
        x = range(len(hybrid_labels))
        width = 0.25

        # 정규화된 점수 사용 (0-1 범위)
        sparse_norm = [r.get('sparse_score_norm', 0) for r in results['hybrid']]
        dense_norm = [r.get('dense_score_norm', 0) for r in results['hybrid']]
        hybrid_scores = [r['hybrid_score'] for r in results['hybrid']]

        ax3.bar([i - width for i in x], sparse_norm, width, label='BM25 (norm)', color='#3498db', alpha=0.8)
        ax3.bar(x, dense_norm, width, label='Semantic (norm)', color='#e74c3c', alpha=0.8)
        ax3.bar([i + width for i in x], hybrid_scores, width, label='Hybrid', color='#2ecc71', alpha=0.8)
        ax3.set_xlabel('Document')
        ax3.set_ylabel('Normalized Score (0-1)')
        ax3.set_title(f'Hybrid Search - Score Fusion (α={alpha})', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(hybrid_labels)
        ax3.set_ylim(0, 1.1)  # 0-1 범위 명확히 표시
        ax3.legend()

        # 4. 문서 출처 정보 (순위 포함)
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = "📊 Search Results Summary\n" + "="*40 + "\n\n"

        info_text += "🔵 Sparse (BM25) - Top Results:\n"
        for i, r in enumerate(results['sparse'][:3], 1):
            source = r['source'][:30] + "..." if len(r['source']) > 30 else r['source']
            info_text += f"  {i}. {source}\n"
            info_text += f"     BM25: {r['score']:.2f}\n"

        info_text += "\n🔴 Dense (Semantic) - Top Results:\n"
        for i, r in enumerate(results['dense'][:3], 1):
            source = r['source'][:30] + "..." if len(r['source']) > 30 else r['source']
            info_text += f"  {i}. {source}\n"
            info_text += f"     L2 Dist: {r['score']:.4f}\n"

        info_text += "\n🟢 Hybrid - Top Results:\n"
        for i, r in enumerate(results['hybrid'][:3], 1):
            source = r['source'][:30] + "..." if len(r['source']) > 30 else r['source']
            s_rank = r.get('sparse_rank', 0)
            d_rank = r.get('dense_rank', 0)
            bm25 = r.get('sparse_score', 0)
            s_norm = r.get('sparse_score_norm', 0)
            d_norm = r.get('dense_score_norm', 0)
            info_text += f"  {i}. {source}\n"
            info_text += f"     Hybrid: {r['hybrid_score']:.2f}\n"
            info_text += f"     BM25={bm25:.1f}({s_norm:.2f}) Sem={d_norm:.2f}\n"

        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = f"./hybrid_search_comparison.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 시각화 저장 완료: {save_path}")
        plt.close()

        return results


# ==================== 대화형 질의응답 ====================
def interactive_qa(rag: RAGSystem, openai_api_key: str = None):
    """대화형 질의응답 모드"""

    # 언어별 메시지
    if rag.language == 'ko':
        print("\n" + "=" * 60)
        print("💬 대화형 질의응답 모드")
        print("=" * 60)
        print("질문을 입력하세요. 종료하려면 'quit', 'exit', 'q'를 입력하세요.")
        prompt_text = "❓ 질문: "
        exit_msg = "👋 질의응답을 종료합니다."
    else:
        print("\n" + "=" * 60)
        print("💬 Interactive Q&A Mode")
        print("=" * 60)
        print("Enter your question. Type 'quit', 'exit', or 'q' to exit.")
        prompt_text = "❓ Question: "
        exit_msg = "👋 Exiting Q&A mode."

    print("-" * 60)

    # OpenAI 클라이언트 초기화 (있는 경우)
    client = None
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
        except:
            pass

    while True:
        try:
            question = input(f"\n{prompt_text}").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q', '종료', '끝']:
                print(f"\n{exit_msg}")
                break

            # 결과 개수 조절
            k = 3
            if 'k=' in question:
                try:
                    k_part = question.split('k=')[1].split()[0]
                    k = int(k_part)
                    question = question.replace(f'k={k_part}', '').strip()
                except:
                    pass

            # 질문 언어 감지
            q_lang = detect_language(question)

            # 한국어 질문인 경우 영어로 번역하여 검색
            search_question = question
            if q_lang == 'ko':
                search_question = translate_to_english(question, openai_api_key)
                if search_question != question:
                    print(f"   🔄 검색어: '{search_question}'")

            result = rag.answer(search_question, k=k)
            result['language'] = q_lang  # 원본 질문의 언어 유지

            # 언어별 출력
            if q_lang == 'ko':
                print(f"\n📚 관련 문서 {len(result['contexts'])}개 검색됨:\n")
            else:
                print(f"\n📚 Found {len(result['contexts'])} relevant documents:\n")

            for i, ctx in enumerate(result['contexts'], 1):
                similarity = 1 / (1 + ctx['score'])

                if q_lang == 'ko':
                    print(f"[{i}] 출처: {ctx['source'][:50]}")
                    print(f"    유사도: {similarity:.2%}")
                else:
                    print(f"[{i}] Source: {ctx['source'][:50]}")
                    print(f"    Similarity: {similarity:.2%}")

                content_preview = ctx['content'].replace('\n', ' ')[:300]
                print(f"    {content_preview}...")
                print("-" * 40)

            # OpenAI로 답변 생성 (API 키가 있는 경우)
            if client:
                try:
                    context_text = "\n\n".join([ctx['content'] for ctx in result['contexts']])

                    if q_lang == 'ko':
                        system_msg = "당신은 의학/과학 논문을 기반으로 질문에 답변하는 전문가입니다. 제공된 문맥을 바탕으로 한국어로 답변해주세요."
                    else:
                        system_msg = "You are an expert answering questions based on medical/scientific papers. Answer based on the provided context."

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )

                    answer = response.choices[0].message.content

                    if q_lang == 'ko':
                        print(f"\n🤖 AI 답변:\n{answer}")
                    else:
                        print(f"\n🤖 AI Answer:\n{answer}")

                except Exception as e:
                    pass

            if q_lang == 'ko':
                print(f"\n📖 참고 문서: {', '.join(result['sources'])}")
            else:
                print(f"\n📖 References: {', '.join(result['sources'])}")

        except KeyboardInterrupt:
            print(f"\n\n{exit_msg}")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# ==================== 메인 실행 ====================
def main():
    print("=" * 60)
    print("🚀 Medical/Scientific Paper RAG System")
    print("   with Paper Summarization & Multilingual Support")
    print("=" * 60)

    # 1. 대화형 설정
    config = Config().interactive_setup()

    # 2. 논문 검색 (영어로 검색)
    print("\n" + "=" * 60)
    print("📚 Step 1: 논문 검색")
    print("=" * 60)

    # 한국어인 경우 번역된 영어 쿼리로 검색
    search_query = config.search_query_en if config.search_query_en else config.search_query
    if config.language == 'ko' and config.search_query_en != config.search_query:
        print(f"   🇰🇷 → 🇺🇸 '{search_query}' (으)로 검색합니다...")

    searcher = PaperSearcher(
        api_key=config.pubmed_api_key,
        email=config.pubmed_email
    )
    papers = searcher.search(
        query=search_query,
        source=config.search_source,
        max_results=config.max_results
    )

    print(f"\n📊 검색 결과: 총 {len(papers)}개 논문")

    if papers:
        print("\n📋 검색된 논문 목록:\n")
        for i, paper in enumerate(papers, 1):
            print(f"[{i}] {paper['source']} | {paper['title'][:65]}...")
            print(f"    저자: {', '.join(paper['authors'][:3])}")
            print(f"    발행: {paper['published']}")
            print()

    if not papers:
        print("❌ 논문을 찾을 수 없습니다.")
        return

    # 3. PDF 다운로드
    print("\n" + "=" * 60)
    print("📥 Step 2: PDF 다운로드")
    print("=" * 60)

    downloader = PDFDownloader(PAPERS_DIR)
    downloaded_files = downloader.download_all(papers)

    if not downloaded_files:
        print("❌ 다운로드된 파일이 없습니다.")
        return

    # 4. 텍스트 추출
    print("\n" + "=" * 60)
    print("📄 Step 3: 텍스트 추출")
    print("=" * 60)

    documents = TextExtractor.extract_all(downloaded_files)

    if not documents:
        print("❌ 추출된 텍스트가 없습니다.")
        return

    # 5. 논문 요약 (OpenAI API 있는 경우)
    if config.openai_api_key:
        summarizer = PaperSummarizer(
            api_key=config.openai_api_key,
            language=config.language
        )
        papers = summarizer.summarize(papers, documents)

    # 6. 임베딩 모델 로드
    print("\n" + "=" * 60)
    print("🧠 Step 4: 임베딩 모델 로드")
    print("=" * 60)

    embeddings = EmbeddingModelFactory.create(
        model_type=config.embedding_model,
        device='cpu',
        openai_api_key=config.openai_api_key
    )

    # 7. RAG 시스템 구축
    print("\n" + "=" * 60)
    print("💾 Step 5: RAG 시스템 구축")
    print("=" * 60)

    rag = RAGSystem(
        embeddings=embeddings,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        language=config.language
    )

    vectorstore = rag.build_vectorstore(documents)
    rag.save_vectorstore(VECTORSTORE_DIR)

    # 8. Hybrid Search 분석 및 시각화
    print("\n" + "=" * 60)
    print("🔀 Step 6: Hybrid Search 분석")
    print("=" * 60)

    hybrid_searcher = HybridSearchSystem(rag, sparse_method=config.sparse_method)

    # 검색어로 3가지 검색 방식 비교
    test_query = search_query  # 원래 검색어 사용
    print(f"\n🔍 검색어: '{test_query}'")
    print("-" * 40)

    search_results = hybrid_searcher.compare_all(test_query, k=5, alpha=0.5)

    # Sparse 방식 이름
    sparse_name = config.sparse_method.upper()

    # 결과 출력
    print(f"\n🔵 Sparse Search ({sparse_name}) 결과:")
    for i, r in enumerate(search_results['sparse'][:3], 1):
        sparse_score = r['score']
        source = r['source'][:50]
        print(f"   [{i}] {sparse_name}: {sparse_score:.2f} | {source}...")

    print(f"\n🔴 Dense Search ({config.embedding_model}) 결과:")
    for i, r in enumerate(search_results['dense'][:3], 1):
        l2_dist = r['score']
        source = r['source'][:50]
        print(f"   [{i}] L2 Dist: {l2_dist:.4f} | {source}...")

    print(f"\n🟢 Hybrid Search 결과 ({sparse_name} + {config.embedding_model}, α=0.5):")
    for i, r in enumerate(search_results['hybrid'][:3], 1):
        sparse_raw = r.get('sparse_score', 0)
        sparse_norm = r.get('sparse_score_norm', 0)
        sem_norm = r.get('dense_score_norm', 0)
        hybrid_score = r.get('hybrid_score', 0)
        source = r['source'][:50]
        print(f"   [{i}] Hybrid: {hybrid_score:.2f} | {sparse_name}={sparse_raw:.1f}({sparse_norm:.2f}) + Semantic({sem_norm:.2f})")
        print(f"       {source}...")

    # 시각화 저장
    print("\n📊 시각화 생성 중...")
    hybrid_searcher.visualize_comparison(test_query, k=5, alpha=0.5)

    # 9. 대화형 질의응답
    interactive_qa(rag, config.openai_api_key)

    print("\n" + "=" * 60)
    print("✅ RAG 시스템 종료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
