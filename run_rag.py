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
from tqdm import tqdm

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
        self.vector_db = 'faiss'  # 'faiss' 또는 'qdrant'
        self.chunk_size = 1000
        self.chunk_overlap = 200
        # Reranking 설정
        self.use_reranking = False
        self.reranker_model = 'ms-marco'  # 'ms-marco', 'ms-marco-large', 'bge-reranker', 'bge-reranker-large'
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

        # Vector DB 선택
        print("\n🗄️ Vector DB 선택:")
        print("   1. FAISS (Facebook AI, 로컬 파일 저장) [기본값]")
        print("   2. Qdrant (Named Vectors, HNSW 인덱스, Payload 필터링)")
        db_choice = input("선택 [1 - FAISS]: ").strip() or "1"
        self.vector_db = {
            '1': 'faiss',
            '2': 'qdrant'
        }.get(db_choice, 'faiss')

        # Reranking 설정
        print("\n🎯 Reranking 설정 (검색 정확도 향상):")
        print("   1. 사용 안함 [기본값]")
        print("   2. MS-MARCO MiniLM (빠름)")
        print("   3. MS-MARCO MiniLM Large (정확)")
        print("   4. BGE Reranker (다국어)")
        print("   5. BGE Reranker Large (고정확도)")
        rerank_choice = input("선택 [1]: ").strip() or "1"
        if rerank_choice == '1':
            self.use_reranking = False
        else:
            self.use_reranking = True
            self.reranker_model = {
                '2': 'ms-marco',
                '3': 'ms-marco-large',
                '4': 'bge-reranker',
                '5': 'bge-reranker-large'
            }.get(rerank_choice, 'ms-marco')

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
        print(f"   🗄️ Vector DB: {self.vector_db.upper()}")
        if self.use_reranking:
            print(f"   🎯 Reranking: {self.reranker_model.upper()}")
        else:
            print(f"   🎯 Reranking: 사용 안함")
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
        print("📝 논문 요약 (OpenAI API)")
        print("=" * 60)
        print(f"   📚 총 {len(papers)}개 논문 요약 예정\n")

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
        success_count = 0
        fail_count = 0

        # tqdm 진행 바로 요약 진행 상황 표시
        with tqdm(
            total=len(papers),
            desc="   🤖 AI 요약 생성",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='cyan'
        ) as pbar:
            for i, paper in enumerate(papers):
                title_short = paper['title'][:35] + "..." if len(paper['title']) > 35 else paper['title']
                pbar.set_postfix_str(f"📄 {title_short}")

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
                    success_count += 1

                except Exception as e:
                    # 실패 시 전체 초록 사용
                    paper['summary'] = f"[초록 원문]\n{paper['abstract']}"
                    fail_count += 1

                summarized_papers.append(paper)
                pbar.update(1)
                time.sleep(0.5)  # API 속도 제한

        # 요약 결과 통계
        print("\n" + "-" * 60)
        print("📊 요약 결과 통계")
        print("-" * 60)
        print(f"   ✅ 요약 성공: {success_count}개")
        if fail_count > 0:
            print(f"   ⚠️ 요약 실패: {fail_count}개 (초록으로 대체)")
        print("-" * 60)

        # 요약 결과 출력
        print("\n" + "=" * 70)
        print("📋 논문 요약 결과")
        print("=" * 70)

        for i, paper in enumerate(summarized_papers):
            print(f"\n{'─' * 70}")
            print(f"📄 [{i+1}/{len(summarized_papers)}] {paper['title']}")
            print(f"{'─' * 70}")
            print(f"👤 저자: {', '.join(paper['authors'][:3])}" + (" 외" if len(paper['authors']) > 3 else ""))
            print(f"📅 발행: {paper['published'][:10] if paper.get('published') else 'N/A'}")
            print(f"📖 출처: {paper['source']}")
            print(f"{'─' * 70}")
            print("📝 요약:")
            print()
            # 요약 내용 전체 출력 (줄바꿈 유지)
            summary = paper.get('summary', 'No summary available')
            for line in summary.split('\n'):
                print(f"   {line}")
            print()

        print("=" * 70)
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


# ==================== 트렌드 분석 클래스 ====================
class TrendAnalyzer:
    """키워드 기반 연구 트렌드 분석"""

    def __init__(self, api_key: str = None, email: str = None, openai_api_key: str = None):
        self.searcher = PaperSearcher(api_key=api_key, email=email)
        self.papers = []
        self.trend_data = {}
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

    def search_papers_for_trend(self, keyword: str, max_results: int = 50, source: str = 'pubmed') -> List[Dict]:
        """트렌드 분석을 위한 논문 검색 (더 많은 결과)"""
        print(f"\n🔍 '{keyword}' 트렌드 분석을 위한 논문 검색 중...")
        self.papers = self.searcher.search(keyword, source=source, max_results=max_results)
        print(f"   📊 총 {len(self.papers)}개 논문 수집")
        return self.papers

    def analyze_publication_trend(self) -> Dict:
        """연도별 출판 트렌드 분석"""
        from collections import Counter

        if not self.papers:
            return {}

        # 연도 추출 및 집계
        years = []
        for paper in self.papers:
            pub_date = paper.get('published', '')
            if pub_date:
                try:
                    year = int(pub_date[:4])
                    if 2000 <= year <= 2030:  # 유효한 연도 범위
                        years.append(year)
                except:
                    pass

        year_counts = Counter(years)
        sorted_years = sorted(year_counts.items())

        self.trend_data['year_trend'] = {
            'years': [y[0] for y in sorted_years],
            'counts': [y[1] for y in sorted_years],
            'total': len(years)
        }

        return self.trend_data['year_trend']

    def extract_key_terms(self, top_n: int = 20) -> Dict:
        """논문 초록에서 핵심 키워드 추출"""
        from collections import Counter
        import re

        if not self.papers:
            return {}

        # 불용어 (영어)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'your', 'i', 'me', 'my', 'he', 'she', 'his', 'her', 'which', 'who',
            'whom', 'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'also', 'now', 'here', 'there', 'then', 'once', 'if', 'because',
            'while', 'although', 'though', 'after', 'before', 'since', 'until',
            'about', 'into', 'through', 'during', 'above', 'below', 'between',
            'under', 'again', 'further', 'then', 'once', 'study', 'studies',
            'results', 'conclusion', 'methods', 'background', 'objective',
            'aim', 'purpose', 'introduction', 'discussion', 'however', 'thus',
            'therefore', 'moreover', 'furthermore', 'additionally', 'including',
            'included', 'using', 'used', 'based', 'associated', 'related',
            'compared', 'among', 'within', 'without', 'between', 'across'
        }

        # 모든 초록에서 단어 추출
        all_words = []
        for paper in self.papers:
            abstract = paper.get('abstract', '')
            # 단어 추출 (영문자만)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
            # 불용어 제거
            words = [w for w in words if w not in stopwords]
            all_words.extend(words)

        # 빈도 계산
        word_counts = Counter(all_words)
        top_terms = word_counts.most_common(top_n)

        self.trend_data['key_terms'] = {
            'terms': [t[0] for t in top_terms],
            'counts': [t[1] for t in top_terms]
        }

        return self.trend_data['key_terms']

    def identify_emerging_topics(self) -> List[Dict]:
        """최근 급부상하는 주제 식별"""
        from collections import Counter, defaultdict
        import re

        if not self.papers or len(self.papers) < 5:
            return []

        # 최근 2년 vs 이전 논문 비교
        recent_papers = []
        older_papers = []

        current_year = 2024  # 현재 연도
        for paper in self.papers:
            pub_date = paper.get('published', '')
            try:
                year = int(pub_date[:4])
                if year >= current_year - 1:
                    recent_papers.append(paper)
                else:
                    older_papers.append(paper)
            except:
                pass

        if not recent_papers or not older_papers:
            return []

        # 불용어
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'study', 'studies', 'results', 'patients', 'treatment', 'using'
        }

        def extract_terms(papers):
            words = []
            for paper in papers:
                abstract = paper.get('abstract', '')
                w = re.findall(r'\b[a-zA-Z]{4,}\b', abstract.lower())
                words.extend([x for x in w if x not in stopwords])
            return Counter(words)

        recent_terms = extract_terms(recent_papers)
        older_terms = extract_terms(older_papers)

        # 최근에 급증한 용어 찾기
        emerging = []
        for term, recent_count in recent_terms.most_common(50):
            older_count = older_terms.get(term, 0)
            if older_count == 0:
                growth = float('inf')
            else:
                growth = (recent_count / len(recent_papers)) / (older_count / len(older_papers))

            if growth > 1.5 and recent_count >= 3:
                emerging.append({
                    'term': term,
                    'recent_count': recent_count,
                    'older_count': older_count,
                    'growth_rate': growth if growth != float('inf') else 999
                })

        # 성장률 순 정렬
        emerging.sort(key=lambda x: x['growth_rate'], reverse=True)
        self.trend_data['emerging_topics'] = emerging[:10]

        return self.trend_data['emerging_topics']

    def summarize_research_content(self, keyword: str) -> Dict:
        """연구 내용 요약 생성"""
        if not self.papers:
            return {}

        # 1. 주요 연구 주제 추출 (제목 기반)
        titles = [p.get('title', '') for p in self.papers[:20]]

        # 2. 주요 연구 방향 파악 (초록에서 패턴 추출)
        research_themes = {
            'diagnosis': 0,      # 진단
            'treatment': 0,      # 치료
            'mechanism': 0,      # 메커니즘
            'prevention': 0,     # 예방
            'clinical': 0,       # 임상
            'review': 0,         # 리뷰/개관
            'epidemiology': 0,   # 역학
            'genetics': 0,       # 유전학
        }

        theme_keywords = {
            'diagnosis': ['diagnosis', 'diagnostic', 'detection', 'screening', 'biomarker'],
            'treatment': ['treatment', 'therapy', 'therapeutic', 'drug', 'intervention', 'medication'],
            'mechanism': ['mechanism', 'pathway', 'molecular', 'cellular', 'signaling'],
            'prevention': ['prevention', 'preventive', 'risk factor', 'protective'],
            'clinical': ['clinical', 'trial', 'patient', 'outcome', 'efficacy'],
            'review': ['review', 'overview', 'comprehensive', 'systematic', 'meta-analysis'],
            'epidemiology': ['epidemiology', 'prevalence', 'incidence', 'population'],
            'genetics': ['genetic', 'gene', 'mutation', 'hereditary', 'genomic'],
        }

        for paper in self.papers:
            text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            for theme, keywords in theme_keywords.items():
                for kw in keywords:
                    if kw in text:
                        research_themes[theme] += 1
                        break

        # 상위 연구 주제
        sorted_themes = sorted(research_themes.items(), key=lambda x: x[1], reverse=True)
        top_themes = [(t, c) for t, c in sorted_themes if c > 0][:5]

        # 3. 대표 논문 선정 (최신 + 관련성) - 10개
        representative_papers = self.papers[:10]

        # 4. 연구 내용 요약 텍스트 생성
        summary_text = self._generate_content_summary(keyword, top_themes, representative_papers)

        self.trend_data['content_summary'] = {
            'themes': top_themes,
            'representative_papers': representative_papers,
            'summary_text': summary_text
        }

        return self.trend_data['content_summary']

    def generate_search_summary(self, keyword: str) -> Dict:
        """검색된 논문들에 대한 간단한 요약 생성"""
        if not self.papers:
            return {}

        # 기본 통계
        total_papers = len(self.papers)

        # 연도 범위 계산
        years = []
        for paper in self.papers:
            pub_date = paper.get('published', '')
            if pub_date:
                try:
                    year = int(pub_date[:4])
                    years.append(year)
                except:
                    pass

        year_range = f"{min(years)}-{max(years)}" if years else "N/A"

        # 소스별 분류
        sources = {}
        for paper in self.papers:
            src = paper.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1

        # AI를 사용한 검색 결과 요약 생성
        search_summary_text = self._generate_search_overview(keyword)

        self.trend_data['search_summary'] = {
            'total_papers': total_papers,
            'year_range': year_range,
            'sources': sources,
            'summary_text': search_summary_text
        }

        return self.trend_data['search_summary']

    def _generate_search_overview(self, keyword: str) -> str:
        """검색 결과에 대한 개요 요약 생성"""
        if not self.openai_api_key or len(self.papers) < 2:
            return self._generate_basic_search_overview(keyword)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            print("   🔍 검색 결과 요약 생성 중...")

            # 논문 제목과 초록 수집
            paper_info = []
            for paper in self.papers[:15]:  # 최대 15개 논문 분석
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')[:300]
                paper_info.append(f"제목: {title}\n초록: {abstract}")

            combined_info = "\n\n".join(paper_info)

            prompt = f"""다음은 '{keyword}' 키워드로 검색된 {len(self.papers)}개의 학술 논문 정보입니다.

{combined_info}

위 논문들을 분석하여 다음 형식으로 검색 결과를 요약해주세요:

🔎 **검색 결과 개요**:
('{keyword}'에 대해 어떤 논문들이 검색되었는지 2-3문장으로 설명)

📑 **주요 연구 내용**:
- (검색된 논문들이 다루는 핵심 주제 3-4가지를 bullet point로)

🎯 **연구 초점**:
(이 분야 연구자들이 주로 관심을 가지는 문제나 질문 1-2문장)

한국어로 간결하고 명확하게 작성해주세요."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 학술 논문 검색 결과를 분석하고 요약하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            print("   ✅ 검색 결과 요약 완료!")
            return summary

        except Exception as e:
            print(f"   ⚠️ AI 요약 실패: {str(e)[:50]}")
            return self._generate_basic_search_overview(keyword)

    def _generate_basic_search_overview(self, keyword: str) -> str:
        """기본 검색 결과 개요 생성"""
        if not self.papers:
            return ""

        # 제목에서 공통 주제 추출
        title_words = []
        for paper in self.papers[:10]:
            title = paper.get('title', '').lower()
            words = [w for w in title.split() if len(w) > 4 and w != keyword.lower()]
            title_words.extend(words)

        from collections import Counter
        common_words = Counter(title_words).most_common(5)
        common_topics = [w[0] for w in common_words]

        summary = f"'{keyword}' 검색 결과, 총 {len(self.papers)}개의 논문이 발견되었습니다.\n"
        if common_topics:
            summary += f"주요 관련 주제: {', '.join(common_topics[:3])}"

        return summary

    def _generate_content_summary(self, keyword: str, themes: List, papers: List) -> str:
        """연구 내용 요약 텍스트 생성"""
        # OpenAI API가 있으면 AI 요약 사용
        if self.openai_api_key and len(self.papers) >= 3:
            ai_summary = self._generate_ai_summary(keyword)
            if ai_summary:
                return ai_summary

        # 기본 요약 생성
        summary_parts = []

        # 주요 연구 분야
        if themes:
            theme_names = {
                'diagnosis': '진단/검출',
                'treatment': '치료/약물',
                'mechanism': '메커니즘/경로',
                'prevention': '예방/위험요인',
                'clinical': '임상연구',
                'review': '종합리뷰',
                'epidemiology': '역학연구',
                'genetics': '유전학연구'
            }
            main_themes = [theme_names.get(t[0], t[0]) for t in themes[:3]]
            summary_parts.append(f"'{keyword}' 연구는 주로 {', '.join(main_themes)} 분야에 집중되어 있습니다.")

        # 최신 연구 동향
        if papers:
            recent_titles = [p.get('title', '')[:80] for p in papers[:3]]
            summary_parts.append(f"\n최근 주요 연구: ")
            for i, title in enumerate(recent_titles, 1):
                summary_parts.append(f"  {i}. {title}...")

        return "\n".join(summary_parts)

    def _generate_ai_summary(self, keyword: str) -> Optional[str]:
        """OpenAI를 사용한 AI 요약 생성"""
        if not self.openai_api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            print("   🤖 OpenAI GPT를 사용하여 연구 내용 분석 중...")

            # 논문 초록 수집 (최대 10개로 확대)
            abstracts = []
            for paper in self.papers[:10]:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')[:600]
                year = paper.get('published', '')[:4]
                authors = paper.get('authors', '')[:50] if paper.get('authors') else ''
                abstracts.append(f"[{year}] {title}\n저자: {authors}\n초록: {abstract}")

            combined_text = "\n\n---\n\n".join(abstracts)

            # 연도별 통계 정보 추가
            year_stats = ""
            if 'year_trend' in self.trend_data:
                years = self.trend_data['year_trend']['years'][-5:]
                counts = self.trend_data['year_trend']['counts'][-5:]
                year_stats = f"\n\n연도별 논문 수: " + ", ".join([f"{y}년: {c}편" for y, c in zip(years, counts)])

            prompt = f"""당신은 의학/과학 연구 동향 분석 전문가입니다. 다음은 '{keyword}'에 관한 최근 연구 논문들입니다.

총 분석 논문 수: {len(self.papers)}편{year_stats}

=== 논문 목록 ===
{combined_text}

위 연구들을 종합적으로 분석하여 '{keyword}' 분야의 연구 동향을 다음 형식으로 요약해주세요:

📌 **연구 개요**: (이 분야가 무엇을 다루는지 1-2문장)

🔬 **주요 연구 방향**:
- (최근 연구들이 집중하는 핵심 주제 2-3가지)

💡 **핵심 발견/결과**:
- (주요 연구 결과나 발견 2-3가지)

🔮 **향후 전망**:
- (연구 트렌드 및 향후 방향 1-2문장)

한국어로 명확하고 구체적으로 작성해주세요. 전문 용어는 필요시 간단한 설명을 추가해주세요."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 의학/과학 연구 동향을 분석하는 전문가입니다. 복잡한 연구 내용을 명확하고 이해하기 쉽게 요약합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            print("   ✅ AI 요약 생성 완료!")
            return summary

        except Exception as e:
            print(f"   ⚠️ AI 요약 생성 실패: {str(e)[:80]}")
            print("   📝 기본 요약으로 대체합니다...")
            return None

    def visualize_trends(self, keyword: str, save_path: str = None) -> str:
        """트렌드 시각화"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'📈 Research Trend Analysis: "{keyword}"',
                    fontsize=14, fontweight='bold')

        # 1. 연도별 출판 트렌드
        ax1 = axes[0, 0]
        if 'year_trend' in self.trend_data:
            years = self.trend_data['year_trend']['years']
            counts = self.trend_data['year_trend']['counts']
            colors = plt.cm.Blues([0.3 + 0.7 * i / len(years) for i in range(len(years))])
            bars = ax1.bar(years, counts, color=colors, edgecolor='navy', alpha=0.8)
            ax1.set_xlabel('Year', fontweight='bold')
            ax1.set_ylabel('Number of Publications', fontweight='bold')
            ax1.set_title('📅 Publication Trend by Year', fontweight='bold')
            ax1.set_xticks(years)
            ax1.tick_params(axis='x', rotation=45)
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=9)

        # 2. 핵심 키워드
        ax2 = axes[0, 1]
        if 'key_terms' in self.trend_data:
            terms = self.trend_data['key_terms']['terms'][:10]
            term_counts = self.trend_data['key_terms']['counts'][:10]
            colors = plt.cm.Greens([0.3 + 0.7 * i / len(terms) for i in range(len(terms))])
            bars = ax2.barh(terms[::-1], term_counts[::-1], color=colors[::-1], edgecolor='darkgreen', alpha=0.8)
            ax2.set_xlabel('Frequency', fontweight='bold')
            ax2.set_title('🔑 Top Keywords in Abstracts', fontweight='bold')
            for bar, count in zip(bars, term_counts[::-1]):
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=9)

        # 3. 급부상 주제
        ax3 = axes[1, 0]
        if 'emerging_topics' in self.trend_data and self.trend_data['emerging_topics']:
            emerging = self.trend_data['emerging_topics'][:8]
            e_terms = [e['term'] for e in emerging]
            e_growth = [min(e['growth_rate'], 10) for e in emerging]  # 최대 10으로 제한
            colors = plt.cm.Reds([0.3 + 0.7 * i / len(e_terms) for i in range(len(e_terms))])
            bars = ax3.barh(e_terms[::-1], e_growth[::-1], color=colors[::-1], edgecolor='darkred', alpha=0.8)
            ax3.set_xlabel('Growth Rate', fontweight='bold')
            ax3.set_title('🚀 Emerging Topics (Recent vs Older)', fontweight='bold')
            for bar, growth in zip(bars, e_growth[::-1]):
                ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{growth:.1f}x', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Not enough data\nfor trend analysis',
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('🚀 Emerging Topics', fontweight='bold')

        # 4. 요약 정보
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary = f"📊 Trend Analysis Summary\n{'='*40}\n\n"
        summary += f"🔍 Keyword: {keyword}\n"
        summary += f"📄 Total Papers: {len(self.papers)}\n\n"

        if 'year_trend' in self.trend_data:
            years = self.trend_data['year_trend']['years']
            counts = self.trend_data['year_trend']['counts']
            if years:
                summary += f"📅 Year Range: {min(years)} - {max(years)}\n"
                max_idx = counts.index(max(counts))
                summary += f"📈 Peak Year: {years[max_idx]} ({counts[max_idx]} papers)\n\n"

        if 'key_terms' in self.trend_data:
            top_5 = self.trend_data['key_terms']['terms'][:5]
            summary += f"🔑 Top Keywords:\n"
            for i, term in enumerate(top_5, 1):
                summary += f"   {i}. {term}\n"

        if 'emerging_topics' in self.trend_data and self.trend_data['emerging_topics']:
            summary += f"\n🚀 Emerging Topics:\n"
            for i, e in enumerate(self.trend_data['emerging_topics'][:3], 1):
                summary += f"   {i}. {e['term']} ({e['growth_rate']:.1f}x growth)\n"

        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         alpha=0.9, edgecolor='orange'))

        plt.tight_layout()

        if save_path is None:
            save_path = f"trend_analysis_{keyword.replace(' ', '_')[:20]}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"   📊 트렌드 시각화 저장: {save_path}")
        return save_path

    def generate_trend_report(self, keyword: str) -> str:
        """트렌드 분석 리포트 생성"""
        report = []
        report.append("\n" + "=" * 60)
        report.append(f"📈 '{keyword}' 연구 트렌드 분석 리포트")
        report.append("=" * 60)

        report.append(f"\n📊 분석 대상: {len(self.papers)}개 논문\n")

        # 검색 결과 요약
        if 'search_summary' in self.trend_data and self.trend_data['search_summary']:
            search_data = self.trend_data['search_summary']
            report.append("-" * 60)
            report.append("🔍 검색 결과 요약")
            report.append("-" * 60)

            # 기본 통계
            report.append(f"\n   📚 검색된 논문: {search_data.get('total_papers', 0)}개")
            report.append(f"   📅 연도 범위: {search_data.get('year_range', 'N/A')}")

            # 소스별 분류
            sources = search_data.get('sources', {})
            if sources:
                source_str = ", ".join([f"{k}: {v}개" for k, v in sources.items()])
                report.append(f"   📖 출처: {source_str}")

            # 요약 텍스트
            summary_text = search_data.get('summary_text', '')
            if summary_text:
                report.append("\n" + "-" * 40)
                summary_lines = summary_text.split('\n')
                for line in summary_lines:
                    if line.strip():
                        report.append(f"   {line}")
            report.append("")

        # 연도별 트렌드
        if 'year_trend' in self.trend_data:
            years = self.trend_data['year_trend']['years']
            counts = self.trend_data['year_trend']['counts']
            if years:
                report.append("📅 연도별 출판 동향:")
                for year, count in zip(years[-5:], counts[-5:]):  # 최근 5년
                    bar = "█" * (count * 2)
                    report.append(f"   {year}: {bar} ({count})")

                # 트렌드 분석
                if len(counts) >= 2:
                    recent_avg = sum(counts[-2:]) / 2
                    older_avg = sum(counts[:-2]) / max(len(counts) - 2, 1) if len(counts) > 2 else counts[0]
                    if recent_avg > older_avg * 1.2:
                        report.append(f"\n   📈 트렌드: 상승세 (최근 연구 증가)")
                    elif recent_avg < older_avg * 0.8:
                        report.append(f"\n   📉 트렌드: 하락세 (연구 관심 감소)")
                    else:
                        report.append(f"\n   ➡️ 트렌드: 안정적 (꾸준한 연구)")

        # 핵심 키워드
        if 'key_terms' in self.trend_data:
            report.append("\n🔑 핵심 키워드 (Top 10):")
            terms = self.trend_data['key_terms']['terms'][:10]
            counts = self.trend_data['key_terms']['counts'][:10]
            for i, (term, count) in enumerate(zip(terms, counts), 1):
                report.append(f"   {i:2d}. {term} ({count}회)")

        # 급부상 주제
        if 'emerging_topics' in self.trend_data and self.trend_data['emerging_topics']:
            report.append("\n🚀 급부상 주제 (최근 vs 과거):")
            for i, e in enumerate(self.trend_data['emerging_topics'][:5], 1):
                growth = e['growth_rate']
                if growth == 999:
                    growth_str = "NEW!"
                else:
                    growth_str = f"{growth:.1f}x"
                report.append(f"   {i}. {e['term']} - {growth_str} 성장")

        # 연구 내용 요약
        if 'content_summary' in self.trend_data and self.trend_data['content_summary']:
            summary_data = self.trend_data['content_summary']
            report.append("\n" + "-" * 60)
            report.append("📋 연구 내용 요약")
            report.append("-" * 60)

            # 주요 연구 주제
            if 'themes' in summary_data and summary_data['themes']:
                theme_names = {
                    'diagnosis': '진단/검출',
                    'treatment': '치료/약물',
                    'mechanism': '메커니즘/경로',
                    'prevention': '예방/위험요인',
                    'clinical': '임상연구',
                    'review': '종합리뷰',
                    'epidemiology': '역학연구',
                    'genetics': '유전학연구'
                }
                report.append("\n🔬 주요 연구 분야:")
                for theme, count in summary_data['themes'][:5]:
                    theme_name = theme_names.get(theme, theme)
                    bar = "▓" * min(count, 20)
                    report.append(f"   • {theme_name}: {bar} ({count}편)")

            # 요약 텍스트
            if 'summary_text' in summary_data and summary_data['summary_text']:
                report.append("\n📝 연구 동향 요약:")
                summary_lines = summary_data['summary_text'].split('\n')
                for line in summary_lines:
                    if line.strip():
                        report.append(f"   {line}")

            # 대표 논문 (최근 10개)
            if 'representative_papers' in summary_data and summary_data['representative_papers']:
                report.append("\n📚 대표 논문 (최근 10개):")
                report.append("-" * 50)
                for i, paper in enumerate(summary_data['representative_papers'][:10], 1):
                    title = paper.get('title', '')[:65]
                    year = paper.get('published', paper.get('year', 'N/A'))[:4] if paper.get('published') or paper.get('year') else 'N/A'
                    authors = paper.get('authors', [])
                    if isinstance(authors, list):
                        author_str = ', '.join(authors[:2])
                        if len(authors) > 2:
                            author_str += ' et al.'
                    else:
                        author_str = str(authors)[:40]
                    source = paper.get('source', '')

                    report.append(f"\n   [{i:2d}] {title}...")
                    report.append(f"       📅 {year} | 👤 {author_str}")
                    if source:
                        report.append(f"       📖 {source}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def analyze(self, keyword: str, max_results: int = 50, source: str = 'pubmed') -> str:
        """전체 트렌드 분석 실행"""
        # 1. 논문 검색
        self.search_papers_for_trend(keyword, max_results, source)

        if not self.papers:
            return f"❌ '{keyword}'에 대한 논문을 찾을 수 없습니다."

        # 2. 검색 결과 요약 생성
        print("\n📋 검색 결과 분석 중...")
        self.generate_search_summary(keyword)

        # 3. 분석 수행
        print("📊 트렌드 분석 중...")
        self.analyze_publication_trend()
        self.extract_key_terms()
        self.identify_emerging_topics()

        # 4. 연구 내용 요약 생성
        print("📝 연구 내용 요약 생성 중...")
        self.summarize_research_content(keyword)

        # 5. 시각화
        print("📈 시각화 생성 중...")
        self.visualize_trends(keyword)

        # 6. 리포트 생성
        report = self.generate_trend_report(keyword)
        print(report)

        return report


# ==================== PDFDownloader 클래스 ====================
class PDFDownloader:
    def __init__(self, save_dir: str = PAPERS_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(self, paper: Dict, show_progress: bool = True) -> Optional[str]:
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
        filename = f"{paper['id']}_{safe_title}.pdf"
        filepath = os.path.join(self.save_dir, filename)
        txt_filepath = filepath.replace('.pdf', '.txt')

        if os.path.exists(txt_filepath):
            return txt_filepath
        if os.path.exists(filepath):
            return filepath

        pdf_url = paper.get('pdf_url') or paper.get('pmc_url')

        if not pdf_url:
            if paper['source'] == 'PubMed':
                return self._save_abstract_as_text(paper, filename)
            return None

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
            response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            # Content-Type 확인
            content_type = response.headers.get('content-type', '')
            if 'html' in content_type or 'text' in content_type:
                return self._save_abstract_as_text(paper, filename)

            # 파일 크기 확인
            total_size = int(response.headers.get('content-length', 0))

            # 처음 몇 바이트로 HTML 여부 확인
            first_chunk = next(response.iter_content(chunk_size=5), b'')
            if first_chunk.startswith(b'<html') or first_chunk.startswith(b'<!DOC'):
                return self._save_abstract_as_text(paper, filename)

            # 스트리밍 다운로드 with 진행 바
            with open(filepath, 'wb') as f:
                f.write(first_chunk)  # 첫 청크 기록

                if show_progress and total_size > 0:
                    # 진행 바 표시
                    with tqdm(
                        total=total_size,
                        initial=len(first_chunk),
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"   📥 {filename[:30]}",
                        bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{rate_fmt}]',
                        leave=False
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # 진행 바 없이 다운로드
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            return filepath

        except Exception as e:
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
        total = len(papers)
        if total == 0:
            print("   ⚠️ 다운로드할 논문이 없습니다.")
            return []

        print("\n" + "=" * 60)
        print("📥 논문 다운로드")
        print("=" * 60)
        print(f"   총 {total}개 논문 다운로드 예정\n")

        downloaded = []
        skipped = 0
        pdf_count = 0
        abstract_count = 0
        failed = 0

        # 전체 진행 상황 바
        with tqdm(
            total=total,
            desc="   📚 전체 진행",
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n}/{total} [{elapsed}<{remaining}]',
            colour='green'
        ) as pbar:
            for i, paper in enumerate(papers):
                title = paper.get('title', 'Unknown')[:35]

                # 이미 존재하는지 확인
                safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                filename = f"{paper['id']}_{safe_title}.pdf"
                filepath = os.path.join(self.save_dir, filename)
                txt_filepath = filepath.replace('.pdf', '.txt')

                if os.path.exists(txt_filepath) or os.path.exists(filepath):
                    skipped += 1
                    pbar.set_postfix_str(f"⏭️ 스킵: {title}...")
                    downloaded.append(txt_filepath if os.path.exists(txt_filepath) else filepath)
                else:
                    pbar.set_postfix_str(f"📥 {title}...")
                    result_path = self.download(paper, show_progress=False)

                    if result_path:
                        downloaded.append(result_path)
                        if result_path.endswith('.pdf'):
                            pdf_count += 1
                        else:
                            abstract_count += 1
                    else:
                        failed += 1

                pbar.update(1)
                time.sleep(0.2)

        # 다운로드 결과 요약
        print("\n" + "-" * 60)
        print("📊 다운로드 결과 요약")
        print("-" * 60)
        print(f"   ✅ 총 다운로드: {len(downloaded)}개")
        print(f"      📄 PDF 파일: {pdf_count}개")
        print(f"      📝 초록 저장: {abstract_count}개")
        print(f"      ⏭️ 기존 파일: {skipped}개")
        if failed > 0:
            print(f"      ❌ 실패: {failed}개")
        print("-" * 60)

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
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if len(texts) > batch_size:
            # tqdm 진행 바 사용
            with tqdm(
                total=len(texts),
                desc="   🧠 임베딩 생성",
                bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
                unit="문서",
                colour='cyan'
            ) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    embeddings = self._encode(batch)
                    all_embeddings.extend(embeddings)
                    pbar.update(len(batch))
        else:
            # 적은 수의 문서는 진행 바 없이 처리
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings = self._encode(batch)
                all_embeddings.extend(embeddings)

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

        if len(texts) > batch_size:
            # tqdm 진행 바 사용
            with tqdm(
                total=len(texts),
                desc="   🧠 OpenAI 임베딩",
                bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
                unit="문서",
                colour='green'
            ) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
        else:
            # 적은 수의 문서는 진행 바 없이 처리
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


# ==================== Reranker 클래스 ====================
class Reranker:
    """Cross-Encoder 기반 Reranking 모델"""

    MODELS = {
        'ms-marco': {
            'name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'description': 'MS MARCO 학습 (빠름, 일반용)',
        },
        'ms-marco-large': {
            'name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'description': 'MS MARCO 학습 (정확, 일반용)',
        },
        'bge-reranker': {
            'name': 'BAAI/bge-reranker-base',
            'description': 'BGE Reranker (다국어 지원)',
        },
        'bge-reranker-large': {
            'name': 'BAAI/bge-reranker-large',
            'description': 'BGE Reranker Large (고정확도)',
        },
    }

    def __init__(self, model_type: str = 'ms-marco', device: str = None):
        """
        Reranker 초기화

        Args:
            model_type: 모델 타입 ('ms-marco', 'ms-marco-large', 'bge-reranker', 'bge-reranker-large')
            device: 'cuda', 'mps', 'cpu' 또는 None (자동 감지)
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.model_type = model_type

        model_info = self.MODELS.get(model_type, self.MODELS['ms-marco'])
        model_name = model_info['name']

        print(f"   🔄 Reranker 모델 로딩: {model_name}")
        print(f"   📍 Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        print(f"   ✅ Reranker 로드 완료!")

    def compute_score(self, query: str, document: str) -> float:
        """단일 쿼리-문서 쌍의 관련성 점수 계산"""
        import torch

        inputs = self.tokenizer(
            query, document,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Sigmoid를 적용하여 0-1 범위로 변환
            score = torch.sigmoid(outputs.logits).squeeze().item()

        return score

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        content_key: str = 'content',
        show_progress: bool = True
    ) -> List[Dict]:
        """
        문서 리스트 Reranking

        Args:
            query: 검색 쿼리
            documents: 문서 리스트 (각 문서는 dict)
            top_k: 반환할 상위 문서 수 (None이면 전체 반환)
            content_key: 문서 dict에서 텍스트를 가져올 키
            show_progress: 진행 바 표시 여부

        Returns:
            rerank_score가 추가된 정렬된 문서 리스트
        """
        import torch

        if not documents:
            return []

        # 배치 처리를 위한 준비
        pairs = [(query, doc.get(content_key, '')) for doc in documents]
        scores = []

        batch_size = 8

        if show_progress and len(documents) > batch_size:
            iterator = tqdm(
                range(0, len(pairs), batch_size),
                desc="   🎯 Reranking",
                bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
                colour='yellow',
                total=(len(pairs) + batch_size - 1) // batch_size
            )
        else:
            iterator = range(0, len(pairs), batch_size)

        for i in iterator:
            batch_pairs = pairs[i:i+batch_size]

            inputs = self.tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.sigmoid(outputs.logits).squeeze(-1)

                if batch_scores.dim() == 0:
                    scores.append(batch_scores.item())
                else:
                    scores.extend(batch_scores.cpu().tolist())

        # 결과에 rerank_score 추가
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = score
            doc_copy['original_score'] = doc.get('hybrid_score', doc.get('score', 0))
            reranked_docs.append(doc_copy)

        # rerank_score로 정렬
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k:
            return reranked_docs[:top_k]
        return reranked_docs


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
        print("\n" + "=" * 60)
        print("🔧 벡터 DB 생성")
        print("=" * 60)

        # Step 1: 텍스트 청킹
        print("\n✂️ Step 1: 텍스트 청킹")
        all_chunks = []
        all_metadata = []

        with tqdm(
            total=len(documents),
            desc="   📄 문서 처리",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='yellow'
        ) as pbar:
            for doc in documents:
                chunks = self.text_splitter.split_text(doc['text'])
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'source': doc['source'],
                        'chunk_id': i
                    })
                pbar.set_postfix_str(f"{len(chunks)} 청크")
                pbar.update(1)

        print(f"   📊 총 청크 수: {len(all_chunks)}개")

        if not all_chunks:
            raise ValueError("청킹된 텍스트가 없습니다!")

        # Step 2: 임베딩 생성 및 벡터 DB 생성
        print("\n🧠 Step 2: 임베딩 생성 및 벡터 DB 구축")

        self.vectorstore = FAISS.from_texts(
            texts=all_chunks,
            embedding=self.embeddings,
            metadatas=all_metadata
        )

        print("\n✅ 벡터 DB 생성 완료!")
        print("-" * 60)
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


# ==================== Qdrant Hybrid Search 시스템 ====================
class QdrantHybridSearch:
    """
    Qdrant 기반 Hybrid Search 시스템
    - Named Vectors: dense + sparse 동시 저장
    - Payload Filtering: 메타데이터 기반 필터링
    - HNSW Index: 빠른 ANN 검색
    """

    def __init__(
        self,
        embeddings,
        sparse_encoder=None,
        collection_name: str = "papers",
        use_memory: bool = True,
        qdrant_url: str = None,
        sparse_method: str = 'bm25',
        reranker: 'Reranker' = None,
        use_reranking: bool = False,
        reranker_model: str = 'ms-marco'
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            VectorParams, SparseVectorParams, Distance,
            HnswConfigDiff, OptimizersConfigDiff
        )

        self.embeddings = embeddings
        self.collection_name = collection_name
        self.sparse_method = sparse_method.lower()
        self.chunks = []
        self.splade_encoder = None
        self.reranker = reranker
        self.use_reranking = use_reranking

        # Qdrant 클라이언트 초기화
        if use_memory:
            print("\n🗄️ Qdrant 인메모리 모드로 초기화...")
            self.client = QdrantClient(":memory:")
        else:
            print(f"\n🗄️ Qdrant 서버 연결: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url)

        # Reranker 초기화 (요청 시)
        if use_reranking and self.reranker is None:
            print("\n🎯 Reranker 초기화...")
            self.reranker = Reranker(model_type=reranker_model)

        # Dense 벡터 차원 확인
        test_embedding = self.embeddings.embed_query("test")
        self.dense_dim = len(test_embedding)
        print(f"   📐 Dense 벡터 차원: {self.dense_dim}")

        # Sparse 인코더 초기화
        if self.sparse_method == 'splade':
            self._init_splade_encoder(sparse_encoder)
        else:
            # BM25용 스테머 초기화
            self._init_stemmer()
            print(f"   🔤 Sparse 방식: BM25 (스테밍)")

    def _init_splade_encoder(self, sparse_encoder=None):
        """SPLADE 인코더 초기화"""
        if sparse_encoder is not None:
            self.splade_encoder = sparse_encoder
            print(f"   🔤 Sparse 방식: SPLADE (외부 인코더)")
        else:
            try:
                print(f"   🔤 SPLADE 인코더 로딩 중...")
                self.splade_encoder = SPLADEEncoder(device='cpu')
                print(f"   ✅ SPLADE 인코더 로드 완료!")
            except Exception as e:
                print(f"   ⚠️ SPLADE 로드 실패: {e}")
                print(f"   ⚠️ BM25로 폴백합니다.")
                self.sparse_method = 'bm25'
                self._init_stemmer()

    def _init_stemmer(self):
        """스테머 초기화"""
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
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def _text_to_sparse_vector(self, text: str) -> Dict[int, float]:
        """텍스트를 스파스 벡터로 변환 (BM25 또는 SPLADE)"""
        if self.sparse_method == 'splade' and self.splade_encoder is not None:
            return self._splade_to_sparse_vector(text)
        else:
            return self._bm25_to_sparse_vector(text)

    def _bm25_to_sparse_vector(self, text: str) -> Dict[int, float]:
        """BM25 스타일 스파스 벡터 생성"""
        from collections import Counter
        import math

        tokens = self._tokenize(text)
        token_counts = Counter(tokens)

        sparse_vector = {}
        for token, count in token_counts.items():
            # 해시 함수로 인덱스 생성 (vocabulary 대신)
            idx = hash(token) % 30000  # 30000 차원으로 제한
            if idx < 0:
                idx = -idx
            tf = 1 + math.log(count) if count > 0 else 0
            sparse_vector[idx] = tf

        return sparse_vector

    def _splade_to_sparse_vector(self, text: str) -> Dict[int, float]:
        """SPLADE 인코더를 사용한 스파스 벡터 생성"""
        # SPLADE 인코딩 (token -> weight 딕셔너리 반환)
        splade_result = self.splade_encoder.encode([text])[0]

        # token 문자열을 정수 인덱스로 변환
        sparse_vector = {}
        for token, weight in splade_result.items():
            # 해시 함수로 인덱스 생성
            idx = hash(token) % 30000
            if idx < 0:
                idx = -idx
            # 같은 인덱스에 여러 토큰이 매핑되면 최대값 사용
            if idx in sparse_vector:
                sparse_vector[idx] = max(sparse_vector[idx], weight)
            else:
                sparse_vector[idx] = weight

        return sparse_vector

    def create_collection(self):
        """컬렉션 생성 (Named Vectors + HNSW 인덱스)"""
        from qdrant_client.models import (
            VectorParams, SparseVectorParams, Distance,
            HnswConfigDiff, OptimizersConfigDiff
        )

        # 기존 컬렉션 삭제
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        print(f"\n📦 Qdrant 컬렉션 생성: {self.collection_name}")
        print("   🔧 HNSW 인덱스 설정 중...")

        # Named Vectors로 컬렉션 생성
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                # Dense 벡터 (HNSW 인덱스)
                "dense": VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,              # 연결 수 (높을수록 정확, 느림)
                        ef_construct=100,  # 구축 시 탐색 범위
                        full_scan_threshold=10000
                    )
                )
            },
            sparse_vectors_config={
                # Sparse 벡터
                "sparse": SparseVectorParams()
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000  # 인덱싱 임계값
            )
        )

        print("   ✅ Dense 벡터: HNSW 인덱스 (COSINE)")
        print(f"   ✅ Sparse 벡터: {self.sparse_method.upper()} + Inverted Index")

    def add_documents(self, documents: List[Dict], text_splitter):
        """문서 추가 (청킹 + 임베딩 + 저장)"""
        from qdrant_client.models import PointStruct, SparseVector
        import uuid

        print("\n" + "=" * 60)
        print("🔧 Qdrant 벡터 DB 구축")
        print("=" * 60)

        # Step 1: 청킹
        print("\n✂️ Step 1: 텍스트 청킹")
        all_chunks = []
        all_metadata = []

        with tqdm(
            total=len(documents),
            desc="   📄 문서 처리",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='yellow'
        ) as pbar:
            for doc in documents:
                chunks = text_splitter.split_text(doc['text'])
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'source': doc['source'],
                        'filepath': doc.get('filepath', ''),
                        'chunk_id': i,
                        'chunk_total': len(chunks),
                        'text_length': len(chunk)
                    })
                pbar.set_postfix_str(f"{len(chunks)} 청크")
                pbar.update(1)

        print(f"   📊 총 청크 수: {len(all_chunks)}개")

        if not all_chunks:
            raise ValueError("청킹된 텍스트가 없습니다!")

        self.chunks = [{'content': c, **m} for c, m in zip(all_chunks, all_metadata)]

        # Step 2: Dense 임베딩 생성
        print("\n🧠 Step 2: Dense 임베딩 생성")
        dense_vectors = self.embeddings.embed_documents(all_chunks)

        # Step 3: Sparse 벡터 생성
        sparse_method_name = self.sparse_method.upper()
        print(f"\n🔤 Step 3: Sparse 벡터 생성 ({sparse_method_name})")
        sparse_vectors = []

        with tqdm(
            total=len(all_chunks),
            desc=f"   📊 {sparse_method_name}",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='magenta'
        ) as pbar:
            for chunk in all_chunks:
                sv = self._text_to_sparse_vector(chunk)
                sparse_vectors.append(sv)
                pbar.update(1)

        # Step 4: Qdrant에 저장
        print("\n💾 Step 4: Qdrant 저장")
        points = []

        with tqdm(
            total=len(all_chunks),
            desc="   🔧 포인트 생성",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='blue'
        ) as pbar:
            for i, (chunk, dense_vec, sparse_vec, metadata) in enumerate(
                zip(all_chunks, dense_vectors, sparse_vectors, all_metadata)
            ):
                sparse_indices = list(sparse_vec.keys())
                sparse_values = list(sparse_vec.values())

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload={
                        "text": chunk,
                        **metadata
                    }
                )
                points.append(point)
                pbar.update(1)

        # 배치 업로드
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size

        with tqdm(
            total=total_batches,
            desc="   📤 Qdrant 업로드",
            bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} [{elapsed}<{remaining}]',
            colour='green'
        ) as pbar:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                pbar.update(1)

        print(f"\n✅ Qdrant 저장 완료! ({len(points)}개 벡터)")
        print("-" * 60)

        # 컬렉션 정보 출력
        info = self.client.get_collection(self.collection_name)
        print(f"   📊 컬렉션 상태: {info.status}")
        print(f"   📊 포인트 수: {info.points_count}")

    def dense_search(self, query: str, k: int = 5, filter_conditions: Dict = None) -> List[Dict]:
        """Dense 벡터 검색 (HNSW ANN)"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams

        query_vector = self.embeddings.embed_query(query)

        # 필터 조건 구성
        query_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)

        # 새로운 Qdrant API 사용
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128)
        )

        return [
            {
                'content': r.payload.get('text', ''),
                'source': r.payload.get('source', 'Unknown'),
                'score': r.score,
                'chunk_id': r.payload.get('chunk_id', 0),
                'method': 'dense'
            }
            for r in results.points
        ]

    def sparse_search(self, query: str, k: int = 5, filter_conditions: Dict = None) -> List[Dict]:
        """Sparse 벡터 검색"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

        sparse_vec = self._text_to_sparse_vector(query)
        sparse_indices = list(sparse_vec.keys())
        sparse_values = list(sparse_vec.values())

        # 필터 조건 구성
        query_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)

        # 새로운 Qdrant API 사용
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_indices,
                values=sparse_values
            ),
            using="sparse",
            query_filter=query_filter,
            limit=k,
            with_payload=True
        )

        return [
            {
                'content': r.payload.get('text', ''),
                'source': r.payload.get('source', 'Unknown'),
                'score': r.score,
                'chunk_id': r.payload.get('chunk_id', 0),
                'method': 'sparse'
            }
            for r in results.points
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filter_conditions: Dict = None,
        use_reranking: bool = None,
        rerank_top_k: int = None
    ) -> List[Dict]:
        """
        Hybrid 검색 (Dense + Sparse) + 선택적 Reranking

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: 0.0 = 순수 Sparse, 1.0 = 순수 Dense
            filter_conditions: 필터 조건
            use_reranking: Reranking 사용 여부 (None이면 self.use_reranking 사용)
            rerank_top_k: Reranking할 후보 수 (None이면 k*3)
        """
        import numpy as np

        # Reranking 설정
        apply_reranking = use_reranking if use_reranking is not None else self.use_reranking
        rerank_candidates = rerank_top_k if rerank_top_k else max(k * 3, 20)

        # 더 많은 후보 검색 (reranking 시 더 많이)
        num_candidates = rerank_candidates if apply_reranking else max(k * 3, 20)

        dense_results = self.dense_search(query, k=num_candidates, filter_conditions=filter_conditions)
        sparse_results = self.sparse_search(query, k=num_candidates, filter_conditions=filter_conditions)

        # 점수 정규화를 위한 최대/최소값
        dense_scores = [r['score'] for r in dense_results] if dense_results else [0]
        sparse_scores = [r['score'] for r in sparse_results] if sparse_results else [0]

        dense_max, dense_min = max(dense_scores), min(dense_scores)
        sparse_max, sparse_min = max(sparse_scores), min(sparse_scores)

        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
        sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0

        # 결과 통합
        doc_scores = {}

        for r in dense_results:
            key = r['content'][:100]
            dense_norm = (r['score'] - dense_min) / dense_range
            doc_scores[key] = {
                'content': r['content'],
                'source': r['source'],
                'chunk_id': r['chunk_id'],
                'dense_score': r['score'],
                'dense_norm': dense_norm,
                'sparse_score': 0,
                'sparse_norm': 0
            }

        for r in sparse_results:
            key = r['content'][:100]
            sparse_norm = (r['score'] - sparse_min) / sparse_range

            if key in doc_scores:
                doc_scores[key]['sparse_score'] = r['score']
                doc_scores[key]['sparse_norm'] = sparse_norm
            else:
                doc_scores[key] = {
                    'content': r['content'],
                    'source': r['source'],
                    'chunk_id': r['chunk_id'],
                    'dense_score': 0,
                    'dense_norm': 0,
                    'sparse_score': r['score'],
                    'sparse_norm': sparse_norm
                }

        # Hybrid 점수 계산
        results = []
        for key, data in doc_scores.items():
            hybrid_score = alpha * data['dense_norm'] + (1 - alpha) * data['sparse_norm']
            results.append({
                'content': data['content'],
                'source': data['source'],
                'chunk_id': data['chunk_id'],
                'hybrid_score': hybrid_score,
                'dense_score': data['dense_score'],
                'dense_norm': data['dense_norm'],
                'sparse_score': data['sparse_score'],
                'sparse_norm': data['sparse_norm'],
                'method': 'hybrid'
            })

        # 정렬
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Reranking 적용 (활성화된 경우)
        if apply_reranking and self.reranker is not None:
            # 상위 후보들만 reranking
            candidates = results[:rerank_candidates]
            reranked = self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=k,
                content_key='content',
                show_progress=len(candidates) > 10
            )
            # reranked 결과에 method 업데이트
            for r in reranked:
                r['method'] = 'hybrid+rerank'
            return reranked

        return results[:k]

    def search_with_filter(
        self,
        query: str,
        source_filter: str = None,
        min_chunk_length: int = None,
        k: int = 5,
        search_type: str = 'hybrid'
    ) -> List[Dict]:
        """Payload 필터링을 적용한 검색"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        conditions = []

        if source_filter:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter)
                )
            )

        if min_chunk_length:
            conditions.append(
                FieldCondition(
                    key="text_length",
                    range=Range(gte=min_chunk_length)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        filter_dict = {}
        if source_filter:
            filter_dict['source'] = source_filter

        if search_type == 'dense':
            return self.dense_search(query, k, filter_conditions=filter_dict if filter_dict else None)
        elif search_type == 'sparse':
            return self.sparse_search(query, k, filter_conditions=filter_dict if filter_dict else None)
        else:
            return self.hybrid_search(query, k, filter_conditions=filter_dict if filter_dict else None)

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        info = self.client.get_collection(self.collection_name)
        return {
            'name': self.collection_name,
            'status': str(info.status),
            'points_count': info.points_count,
            'vectors_count': info.vectors_count,
            'indexed_vectors_count': info.indexed_vectors_count
        }

    def get_sparse_method_name(self) -> str:
        """현재 사용 중인 sparse 방식 이름 반환"""
        return self.sparse_method.upper()

    def visualize_comparison(
        self,
        query: str,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        hybrid_results: List[Dict],
        alpha: float = 0.7,
        sparse_method: str = 'BM25',
        dense_model: str = 'Dense',
        save_path: str = None,
        show_plot: bool = False
    ) -> str:
        """Qdrant 검색 결과 시각화 (파일 저장)"""
        import matplotlib
        matplotlib.use('Agg')  # 비대화형 백엔드 (빠르고 안정적)
        import matplotlib.pyplot as plt

        # 한글 폰트 설정
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(f'🔍 Qdrant Hybrid Search Analysis\nQuery: "{query[:60]}..."',
                    fontsize=14, fontweight='bold')

        # 1. Sparse 점수 (왼쪽 상단)
        ax1 = axes[0, 0]
        if sparse_results:
            sparse_labels = [f"Doc {i+1}\n{r['source'][:20]}..." for i, r in enumerate(sparse_results[:5])]
            sparse_scores = [r['score'] for r in sparse_results[:5]]
            colors1 = plt.cm.Blues([(0.4 + 0.12*i) for i in range(len(sparse_scores))])
            bars1 = ax1.barh(sparse_labels, sparse_scores, color=colors1, edgecolor='navy', alpha=0.85)
            ax1.set_xlabel(f'{sparse_method} Score', fontweight='bold')
            ax1.set_title(f'🔵 Sparse Search ({sparse_method})', fontweight='bold', fontsize=11)
            ax1.invert_yaxis()
            for bar, score in zip(bars1, sparse_scores):
                ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
            ax1.set_xlim(0, max(sparse_scores) * 1.3 if sparse_scores else 1)

        # 2. Dense 점수 (오른쪽 상단)
        ax2 = axes[0, 1]
        if dense_results:
            dense_labels = [f"Doc {i+1}\n{r['source'][:20]}..." for i, r in enumerate(dense_results[:5])]
            dense_scores = [r['score'] for r in dense_results[:5]]
            colors2 = plt.cm.Reds([(0.4 + 0.12*i) for i in range(len(dense_scores))])
            bars2 = ax2.barh(dense_labels, dense_scores, color=colors2, edgecolor='darkred', alpha=0.85)
            ax2.set_xlabel('Cosine Similarity (higher = better)', fontweight='bold')
            ax2.set_title(f'🔴 Dense Search (HNSW, {dense_model})', fontweight='bold', fontsize=11)
            ax2.invert_yaxis()
            for bar, score in zip(bars2, dense_scores):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.4f}', va='center', fontsize=9, fontweight='bold')
            ax2.set_xlim(0, max(dense_scores) * 1.2 if dense_scores else 1)

        # 3. Hybrid 점수 비교 (왼쪽 하단)
        ax3 = axes[1, 0]
        if hybrid_results:
            hybrid_labels = [f"Doc {i+1}" for i in range(len(hybrid_results[:5]))]
            x = range(len(hybrid_labels))
            width = 0.25

            sparse_norm = [r.get('sparse_norm', 0) for r in hybrid_results[:5]]
            dense_norm = [r.get('dense_norm', 0) for r in hybrid_results[:5]]
            hybrid_scores = [r['hybrid_score'] for r in hybrid_results[:5]]

            bars_sparse = ax3.bar([i - width for i in x], sparse_norm, width,
                                 label=f'{sparse_method} (norm)', color='#3498db', alpha=0.85, edgecolor='navy')
            bars_dense = ax3.bar(x, dense_norm, width,
                                label='Dense (norm)', color='#e74c3c', alpha=0.85, edgecolor='darkred')
            bars_hybrid = ax3.bar([i + width for i in x], hybrid_scores, width,
                                 label='Hybrid', color='#2ecc71', alpha=0.85, edgecolor='darkgreen')

            ax3.set_xlabel('Document', fontweight='bold')
            ax3.set_ylabel('Normalized Score (0-1)', fontweight='bold')
            ax3.set_title(f'🟢 Hybrid Score Fusion (α={alpha})\n{sparse_method}×{1-alpha:.1f} + Dense×{alpha:.1f}',
                         fontweight='bold', fontsize=11)
            ax3.set_xticks(x)
            ax3.set_xticklabels(hybrid_labels)
            ax3.set_ylim(0, 1.15)
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(axis='y', alpha=0.3)

            # 하이브리드 점수 표시
            for bar, score in zip(bars_hybrid, hybrid_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 4. 상세 결과 요약 (오른쪽 하단)
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = "📊 Qdrant Search Results Summary\n" + "="*45 + "\n\n"

        info_text += f"🔵 Sparse ({sparse_method}) - Top 3:\n"
        for i, r in enumerate(sparse_results[:3], 1):
            source = r['source'][:35] + "..." if len(r['source']) > 35 else r['source']
            info_text += f"  {i}. {source}\n"
            info_text += f"     Score: {r['score']:.4f}\n"

        info_text += f"\n🔴 Dense (HNSW) - Top 3:\n"
        for i, r in enumerate(dense_results[:3], 1):
            source = r['source'][:35] + "..." if len(r['source']) > 35 else r['source']
            info_text += f"  {i}. {source}\n"
            info_text += f"     Cosine: {r['score']:.4f}\n"

        info_text += f"\n🟢 Hybrid (α={alpha}) - Top 3:\n"
        for i, r in enumerate(hybrid_results[:3], 1):
            source = r['source'][:35] + "..." if len(r['source']) > 35 else r['source']
            sparse_s = r.get('sparse_score', 0)
            dense_s = r.get('dense_score', 0)
            hybrid_s = r.get('hybrid_score', 0)
            info_text += f"  {i}. {source}\n"
            info_text += f"     Hybrid: {hybrid_s:.3f} (S:{sparse_s:.3f} D:{dense_s:.3f})\n"

        info_text += f"\n📈 Statistics:\n"
        info_text += f"  • Collection: {self.collection_name}\n"
        info_text += f"  • Total chunks: {len(self.chunks)}\n"
        info_text += f"  • Dense dim: {self.dense_dim}\n"

        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         alpha=0.9, edgecolor='orange'))

        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = f"qdrant_hybrid_search_{query[:20].replace(' ', '_')}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # 명시적으로 figure 닫기

        print(f"   📊 시각화 저장 완료: {save_path}")
        return save_path


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

    def __init__(
        self,
        rag_system: RAGSystem,
        sparse_method: str = 'bm25',
        use_reranking: bool = False,
        reranker_model: str = 'ms-marco'
    ):
        """
        Args:
            rag_system: RAG 시스템
            sparse_method: 'bm25' 또는 'splade'
            use_reranking: Reranking 사용 여부
            reranker_model: Reranker 모델 타입
        """
        from rank_bm25 import BM25Okapi
        import numpy as np

        self.rag = rag_system
        self.chunks = rag_system.get_all_chunks()
        self.np = np
        self.sparse_method = sparse_method.lower()
        self.use_reranking = use_reranking
        self.reranker = None

        # Reranker 초기화 (요청 시)
        if use_reranking:
            print("\n🎯 Reranker 초기화...")
            self.reranker = Reranker(model_type=reranker_model)

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

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 10,
        use_reranking: bool = None,
        rerank_top_k: int = None
    ) -> List[Dict]:
        """
        Hybrid 검색 (Sparse + Dense 결합) - RRF (Reciprocal Rank Fusion) 사용 + Reranking

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: 0.0 = 순수 Sparse, 1.0 = 순수 Dense
            rrf_k: RRF 상수 (기본값 10)
            use_reranking: Reranking 사용 여부 (None이면 self.use_reranking)
            rerank_top_k: Reranking할 후보 수
        """
        # Reranking 설정
        apply_reranking = use_reranking if use_reranking is not None else self.use_reranking
        rerank_candidates = rerank_top_k if rerank_top_k else max(k * 3, 20)

        # 충분한 후보 검색
        num_candidates = rerank_candidates if apply_reranking else max(k * 3, 20)
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

        # Reranking 적용 (활성화된 경우)
        if apply_reranking and self.reranker is not None:
            candidates = results[:rerank_candidates]
            reranked = self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=k,
                content_key='content',
                show_progress=len(candidates) > 10
            )
            for r in reranked:
                r['method'] = 'hybrid+rerank'
            return reranked

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
        """검색 결과 시각화 (파일 저장)"""
        import matplotlib
        matplotlib.use('Agg')  # 비대화형 백엔드 (빠르고 안정적)
        import matplotlib.pyplot as plt

        # 한글 폰트 설정
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
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

        plt.savefig(save_path, dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # 명시적으로 figure 닫기

        print(f"   📊 시각화 저장 완료: {save_path}")
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


# ==================== 트렌드 분석 실행 ====================
def run_trend_analysis():
    """트렌드 분석 모드 실행"""
    print("\n" + "=" * 60)
    print("📈 연구 트렌드 분석")
    print("=" * 60)

    # .env에서 API 키 로드
    pubmed_api_key = os.getenv('PUBMED_API_KEY')
    pubmed_email = os.getenv('PUBMED_EMAIL')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        print("   ✅ OpenAI API 연결됨 - AI 요약 기능 활성화")
    else:
        print("   ⚠️ OpenAI API 미설정 - 기본 요약 사용")

    # 검색 소스 선택
    print("\n📖 검색 소스 선택:")
    print("   1. PubMed (의학/생물학) [기본값]")
    print("   2. arXiv (CS/물리/수학)")
    print("   3. 둘 다")
    source_choice = input("선택 [1]: ").strip() or "1"
    source = {'1': 'pubmed', '2': 'arxiv', '3': 'both'}.get(source_choice, 'pubmed')

    # 키워드 입력
    keyword = input("\n🔍 분석할 키워드 입력: ").strip()
    if not keyword:
        keyword = "diabetes treatment"
        print(f"   기본값 사용: {keyword}")

    # 언어 감지 및 번역
    if detect_language(keyword) == 'ko':
        openai_api_key = os.getenv('OPENAI_API_KEY')
        print("\n🔄 한국어 키워드를 영어로 번역 중...")
        keyword_en = translate_to_english(keyword, openai_api_key)
        print(f"   🇰🇷 원본: {keyword}")
        print(f"   🇺🇸 번역: {keyword_en}")
        keyword = keyword_en

    # 논문 수 설정
    max_results = input("\n📄 분석할 최대 논문 수 [50]: ").strip()
    max_results = int(max_results) if max_results.isdigit() else 50

    print("\n" + "-" * 60)
    print("✅ 트렌드 분석 설정 완료!")
    print(f"   🔍 키워드: {keyword}")
    print(f"   📖 소스: {source}")
    print(f"   📄 최대 논문: {max_results}")
    print("-" * 60)

    # 트렌드 분석 실행
    analyzer = TrendAnalyzer(api_key=pubmed_api_key, email=pubmed_email, openai_api_key=openai_api_key)
    analyzer.analyze(keyword, max_results=max_results, source=source)

    # 추가 분석 여부
    while True:
        another = input("\n🔄 다른 키워드 분석? (y/n) [n]: ").strip().lower()
        if another == 'y':
            keyword = input("🔍 새 키워드 입력: ").strip()
            if keyword:
                if detect_language(keyword) == 'ko':
                    keyword = translate_to_english(keyword, os.getenv('OPENAI_API_KEY'))
                    print(f"   🇺🇸 번역: {keyword}")
                analyzer.analyze(keyword, max_results=max_results, source=source)
        else:
            break

    print("\n" + "=" * 60)
    print("✅ 트렌드 분석 종료!")
    print("=" * 60)


# ==================== 메인 실행 ====================
def main():
    print("=" * 60)
    print("🚀 Medical/Scientific Paper RAG System")
    print("   with Paper Summarization & Multilingual Support")
    print("=" * 60)

    # 메인 메뉴
    print("\n📋 기능 선택:")
    print("   1. RAG 시스템 (논문 검색 + 질의응답)")
    print("   2. 트렌드 분석 (키워드 기반 연구 동향)")
    mode = input("선택 [1]: ").strip() or "1"

    if mode == "2":
        # 트렌드 분석 모드
        run_trend_analysis()
        return

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

    # Text Splitter 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len
    )

    # Vector DB 선택에 따라 분기
    if config.vector_db == 'qdrant':
        # Qdrant 기반 시스템
        print(f"\n🗄️ Qdrant Vector DB 사용")

        qdrant_search = QdrantHybridSearch(
            embeddings=embeddings,
            sparse_method=config.sparse_method,
            collection_name="medical_papers",
            use_memory=True  # 인메모리 모드
        )
        qdrant_search.create_collection()
        qdrant_search.add_documents(documents, text_splitter)

        # RAG 시스템도 생성 (interactive_qa용)
        rag = RAGSystem(
            embeddings=embeddings,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            language=config.language
        )
        rag.build_vectorstore(documents)

        # 8. Hybrid Search 분석 및 시각화 (Qdrant)
        print("\n" + "=" * 60)
        print("🔀 Step 6: Qdrant Hybrid Search 분석")
        print("=" * 60)

        test_query = search_query
        print(f"\n🔍 검색어: '{test_query}'")
        print("-" * 40)

        # Qdrant 검색 실행
        sparse_results = qdrant_search.sparse_search(test_query, k=5)
        dense_results = qdrant_search.dense_search(test_query, k=5)
        hybrid_results = qdrant_search.hybrid_search(test_query, k=5, alpha=0.7)

        # 실제 사용 중인 sparse 방식 (SPLADE 로드 실패 시 BM25로 폴백될 수 있음)
        sparse_name = qdrant_search.get_sparse_method_name()

        # Sparse 결과
        print(f"\n🔵 Qdrant Sparse Search ({sparse_name}) 결과:")
        for i, r in enumerate(sparse_results[:3], 1):
            score = r['score']
            source = r['source'][:50]
            print(f"   [{i}] {sparse_name}: {score:.4f} | {source}...")

        # Dense 결과 (HNSW)
        print(f"\n🔴 Qdrant Dense Search (HNSW, {config.embedding_model}) 결과:")
        for i, r in enumerate(dense_results[:3], 1):
            score = r['score']
            source = r['source'][:50]
            print(f"   [{i}] Cosine: {score:.4f} | {source}...")

        # Hybrid 결과
        print(f"\n🟢 Qdrant Hybrid Search 결과 ({sparse_name} + Dense, α=0.7):")
        for i, r in enumerate(hybrid_results[:3], 1):
            hybrid_score = r.get('hybrid_score', 0)
            sparse_score = r.get('sparse_score', 0)
            dense_score = r.get('dense_score', 0)
            source = r['source'][:50]
            print(f"   [{i}] Hybrid: {hybrid_score:.3f} | Sparse={sparse_score:.3f} + Dense={dense_score:.3f}")
            print(f"       {source}...")

        # Payload 필터링 예시
        print(f"\n🔎 Payload 필터링 테스트:")
        filtered_results = qdrant_search.search_with_filter(
            test_query,
            min_chunk_length=100,
            k=3
        )
        print(f"   (최소 청크 길이 100자 이상 필터)")
        for i, r in enumerate(filtered_results[:3], 1):
            print(f"   [{i}] {r['source'][:50]}...")

        # 시각화 생성
        print("\n📊 시각화 생성 중...")
        qdrant_search.visualize_comparison(
            query=test_query,
            sparse_results=sparse_results,
            dense_results=dense_results,
            hybrid_results=hybrid_results,
            alpha=0.7,
            sparse_method=sparse_name,
            dense_model=config.embedding_model
        )

    else:
        # FAISS 기반 시스템 (기존)
        print(f"\n🗄️ FAISS Vector DB 사용")

        rag = RAGSystem(
            embeddings=embeddings,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            language=config.language
        )

        vectorstore = rag.build_vectorstore(documents)
        rag.save_vectorstore(VECTORSTORE_DIR)

        # 8. Hybrid Search 분석 및 시각화 (FAISS)
        print("\n" + "=" * 60)
        print("🔀 Step 6: Hybrid Search 분석")
        print("=" * 60)

        hybrid_searcher = HybridSearchSystem(rag, sparse_method=config.sparse_method)

        test_query = search_query
        print(f"\n🔍 검색어: '{test_query}'")
        print("-" * 40)

        search_results = hybrid_searcher.compare_all(test_query, k=5, alpha=0.5)

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

        # 시각화 저장 및 표시
        print("\n📊 시각화 생성 중...")
        hybrid_searcher.visualize_comparison(test_query, k=5, alpha=0.5)

    # 9. 대화형 질의응답
    interactive_qa(rag, config.openai_api_key)

    print("\n" + "=" * 60)
    print("✅ RAG 시스템 종료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
