#!/usr/bin/env python3
"""
RAG 워크플로우 테스트 스크립트
1. 첫 번째 검색어로 논문 검색 → 임베딩 → VectorDB 저장
2. 두 번째 검색어로 VectorDB에서 관련 내용 검색 → 출력
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# run_rag.py에서 필요한 클래스와 함수 임포트
from run_rag import (
    Config, PaperSearcher, RAGSystem,
    EmbeddingModelFactory, detect_language, translate_to_english
)

def test_rag_workflow():
    """RAG 워크플로우 테스트"""

    print("=" * 70)
    print("🧪 RAG 워크플로우 테스트")
    print("=" * 70)

    # 1. 설정
    print("\n📋 Step 1: 설정 초기화")
    print("-" * 50)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    pubmed_api_key = os.getenv('PUBMED_API_KEY')
    pubmed_email = os.getenv('PUBMED_EMAIL')

    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    else:
        print("✅ OpenAI API 연결됨")

    # 2. 첫 번째 검색어로 논문 검색
    first_query = "COVID-19 vaccine efficacy"
    print(f"\n📚 Step 2: 첫 번째 검색어로 논문 검색")
    print("-" * 50)
    print(f"   🔍 검색어: '{first_query}'")

    searcher = PaperSearcher(api_key=pubmed_api_key, email=pubmed_email)
    papers = searcher.search(query=first_query, source='pubmed', max_results=5)

    print(f"   📊 검색된 논문: {len(papers)}개")
    for i, paper in enumerate(papers, 1):
        print(f"   [{i}] {paper['title'][:60]}...")

    if not papers:
        print("❌ 논문을 찾을 수 없습니다. 테스트 중단.")
        return False

    # 3. 임베딩 모델 로드
    print(f"\n🧠 Step 3: 임베딩 모델 로드")
    print("-" * 50)

    embeddings = EmbeddingModelFactory.create(
        model_type='pubmedbert',
        device='cpu',
        openai_api_key=openai_api_key
    )
    print("   ✅ PubMedBERT 임베딩 모델 로드 완료")

    # 4. RAG 시스템 구축 (Abstract 기반)
    print(f"\n💾 Step 4: VectorDB 구축 (Abstract 기반)")
    print("-" * 50)

    rag = RAGSystem(
        embeddings=embeddings,
        chunk_size=500,
        chunk_overlap=100,
        language='en'
    )
    rag.build_vectorstore_from_abstracts(papers)
    print("   ✅ VectorDB 구축 완료")

    # 5. 저장된 청크 확인
    chunks = rag.get_all_chunks()
    print(f"   📊 저장된 청크 수: {len(chunks)}개")

    # 6. 두 번째 검색어로 VectorDB 검색
    second_query = "vaccine side effects"
    print(f"\n🔎 Step 5: 두 번째 검색어로 VectorDB 검색")
    print("-" * 50)
    print(f"   🔍 검색어: '{second_query}'")

    results = rag.search(second_query, k=3)

    print(f"\n   📚 검색 결과: {len(results)}개 문서 검색됨")
    print("-" * 50)

    for i, result in enumerate(results, 1):
        similarity = 1 / (1 + result['score'])
        print(f"\n   [{i}] 유사도: {similarity:.2%}")
        print(f"       출처: {result['source'][:50]}...")
        print(f"       내용: {result['content'][:200]}...")

    # 7. 세 번째 검색어 테스트 (한국어)
    third_query = "백신 효과"
    print(f"\n🔎 Step 6: 세 번째 검색어 테스트 (한국어)")
    print("-" * 50)
    print(f"   🔍 원본 검색어: '{third_query}'")

    # 한국어 → 영어 번역
    translated_query = translate_to_english(third_query, openai_api_key)
    print(f"   🔄 번역된 검색어: '{translated_query}'")

    results3 = rag.search(translated_query, k=3)

    print(f"\n   📚 검색 결과: {len(results3)}개 문서 검색됨")

    for i, result in enumerate(results3, 1):
        similarity = 1 / (1 + result['score'])
        print(f"\n   [{i}] 유사도: {similarity:.2%}")
        print(f"       출처: {result['source'][:50]}...")
        print(f"       내용: {result['content'][:150]}...")

    # 8. 테스트 결과 요약
    print("\n" + "=" * 70)
    print("✅ RAG 워크플로우 테스트 완료!")
    print("=" * 70)
    print(f"""
    📊 테스트 결과 요약:
    ─────────────────────────────────────
    • 첫 번째 검색어: '{first_query}'
      → 논문 {len(papers)}개 검색 → 임베딩 → VectorDB 저장

    • 두 번째 검색어: '{second_query}'
      → VectorDB에서 {len(results)}개 관련 문서 검색

    • 세 번째 검색어: '{third_query}' (한국어)
      → 번역: '{translated_query}'
      → VectorDB에서 {len(results3)}개 관련 문서 검색
    ─────────────────────────────────────
    """)

    return True


if __name__ == "__main__":
    success = test_rag_workflow()
    sys.exit(0 if success else 1)
