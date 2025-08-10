#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: search.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import os
from urllib.parse import urlparse
from typing import Optional

from ddgs import DDGS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class SourceService(object):
    def __init__(self, config):
        self.vector_store = None
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.docs_path = self.config.docs_path
        self.vector_store_path = self.config.vector_store_path

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        docs = []
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                print(doc)
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                docs.extend(doc)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)

    def add_document(self, document_path):
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized. Call init_source_vector() or load_vector_store() first.")
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self.vector_store

    def search_web(self, query, *, news: bool = True, max_results: int = 8, timelimit: str = 'w'):
        """基于 ddgs 的网页/新闻检索，并做域名白名单+垃圾词过滤。

        参数:
        - news: True 使用新闻聚合，False 使用通用网页搜索
        - max_results: 返回条数上限
        - timelimit: d(天)/w(周)/m(月)/y(年)，仅对新闻更有意义
        """
        # 优先使用环境变量 DDGS_PROXY，其次回退到本地代理端口
        proxy = os.environ.get('DDGS_PROXY') or 'http://localhost:10809'
        region = 'cn-zh'  # 使用 country-language 规范，避免引擎解析错误
        backend = 'duckduckgo,yahoo,yandex'  # 收敛为较稳定的后端，避开易失败的 google/brave
        safesearch = 'on'

        whitelist_domains = {
            'gov.cn', 'court.gov.cn', 'chinacourt.gov.cn', 'chinacourt.org',
            'legaldaily.com.cn', 'news.cn', 'xinhuanet.com', 'people.com.cn',
            'cctv.com', 'cctv.cn', 'ce.cn', 'china.com.cn', 'thepaper.cn',
            'caixin.com', 'moj.gov.cn', 'spp.gov.cn', 'npc.gov.cn', 'scio.gov.cn'
        }
        blacklist_keywords = ['吃瓜', '黑料', '成人', '色情', '赌博', '博彩', '贷款', '18+']

        def in_whitelist(url: str) -> bool:
            try:
                host = urlparse(url).netloc.lower()
            except Exception:
                return False
            return any(host.endswith(d) for d in whitelist_domains)

        def contains_blacklist(text: Optional[str]) -> bool:
            if not text:
                return False
            return any(k in text for k in blacklist_keywords)

        try:
            with DDGS(proxy=proxy) as ddgs:
                if news:
                    results = ddgs.news(
                        query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        max_results=max_results,
                        backend=backend,
                    )
                else:
                    results = ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=max_results,
                        backend=backend,
                    )

            if not results:
                return ''

            # 先按白名单+黑词过滤
            filtered: list[tuple[str, str, str]] = []
            for r in results:
                title = (r.get('title') or '').strip()
                url = r.get('href') or r.get('url') or r.get('embed_url') or r.get('image') or ''
                body = (r.get('body') or r.get('snippet') or r.get('excerpt') or '').strip()
                if not url:
                    continue
                if contains_blacklist(title) or contains_blacklist(body):
                    continue
                if in_whitelist(url):
                    filtered.append((title, url, body))

            # 白名单为空则退化为仅黑词过滤的前若干条
            if not filtered:
                for r in results:
                    title = (r.get('title') or '').strip()
                    url = r.get('href') or r.get('url') or r.get('embed_url') or r.get('image') or ''
                    body = (r.get('body') or r.get('snippet') or r.get('excerpt') or '').strip()
                    if not url:
                        continue
                    if contains_blacklist(title) or contains_blacklist(body):
                        continue
                    filtered.append((title, url, body))
                    if len(filtered) >= max_results:
                        break

            # 组装文本结果
            lines: list[str] = []
            for i, (title, url, body) in enumerate(filtered[:max_results], start=1):
                snippet = body
                if len(snippet) > 200:
                    snippet = snippet[:200] + '…'
                lines.append(f'[{i}] {title}\n{url}\n{snippet}')

            return '\n\n'.join(lines)
        except Exception as e:
            print(f"网络检索异常:{query}: {e}")
            return ''

# if __name__ == '__main__':
#     config = LangChainCFG()
#     source_service = SourceService(config)
#     source_service.init_source_vector()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/科比.txt')
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     vector_store=source_service.load_vector_store()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)

if __name__ == '__main__':
    from main import LangChainCFG
    config = LangChainCFG()
    web =  SourceService(config).search_web('请搜索最新的法律相关新闻')
    print(web)