import os
import shutil

from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.langchain_application import LangChainApplication

# 环境变量，使得代码只能访问序号2,3的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/user/yuanjinmin/LexiLaw/demo
# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = current_dir + '/model'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = current_dir + '/text2vec'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = current_dir + '/cache'
    docs_path = current_dir + '/docs'
    kg_vector_stores = {
        '刑法法条': current_dir + '/cache/legal_article',
        '刑法书籍': current_dir + '/cache/legal_book',
        '初始化': current_dir + '/cache',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    n_gpus=1

if __name__ == '__main__':
    config = LangChainCFG()
    application = LangChainApplication(config)

    # application.source_service.init_source_vector()

    def get_file_list():
        if not os.path.exists("docs"):
            return []
        return [f for f in os.listdir("docs")]


    file_list = get_file_list()


    def upload_file(file):
        if not os.path.exists("docs"):
            os.mkdir("docs")
        filename = os.path.basename(file.name)
        shutil.move(file.name, "docs/" + filename)
        # file_list首位插入新上传的文件
        file_list.insert(0, filename)
        application.source_service.add_document("docs/" + filename)
        return gr.Dropdown.update(choices=file_list, value=filename)


    def set_knowledge(kg_name, history):
        try:
            application.source_service.load_vector_store(config.kg_vector_stores[kg_name])
            msg_status = f'{kg_name}知识库已成功加载'
        except Exception as e:
            print(e)
            msg_status = f'{kg_name}知识库未成功加载'
        return history + [[None, msg_status]]


    def clear_session():
        return '', None


    def predict(input,
                large_language_model,
                embedding_model,
                top_k,
                use_web,
                use_pattern,
                history=None):
        # print(large_language_model, embedding_model)
        print(input)
        if history == None:
            history = []

        if use_web == '使用':
            web_content = application.source_service.search_web(query=input)
        else:
            web_content = ''
        search_text = ''
        if use_pattern == '模型问答':
            result = application.get_llm_answer(query=input, web_content=web_content)
            history.append((input, result))
            search_text += web_content
            return '', history, history, search_text

        else:
            resp = application.get_knowledge_based_answer(
                query=input,
                history_len=1,
                temperature=0.1,
                top_p=0.9,
                top_k=top_k,
                web_content=web_content,
                chat_history=history
            )
            history.append((input, resp['result']))
            for idx, source in enumerate(resp['source_documents'][:4]):
                sep = f'----------【搜索结果{idx + 1}：】---------------\n'
                search_text += f'{sep}\n{source.page_content}\n\n'
            print(search_text)
            search_text += "----------【网络检索内容】-----------\n"
            search_text += web_content
            return '', history, history, search_text

    with open(os.path.join(current_dir, "assets/custom.css"), "r", encoding="utf-8") as f:
        customCSS = f.read()
    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
        gr.Markdown("""<h1><center>基于法律知识库和大语言模型的法律助手</center></h1>
            <center><font size=3>
            </center></font>
            """)
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                embedding_model = gr.Dropdown([
                    "text2vec-base"
                ],
                    label="嵌入模型",
                    value="text2vec-base")

                large_language_model = gr.Dropdown(
                    [
                        "ChatGLM-6B",
                    ],
                    label="大语言模型",
                    value="ChatGLM-6B")

                top_k = gr.Slider(1,
                                20,
                                value=4,
                                step=1,
                                label="检索top-k文档",
                                interactive=True)

                use_web = gr.Radio(["使用", "不使用"], label="联网搜索",
                                info="是否使用网络搜索，使用时确保网络通畅",
                                value="不使用"
                                )
                use_pattern = gr.Radio(
                    [
                        '模型问答',
                        '知识库问答',
                    ],
                    label="模式",
                    value='模型问答',
                    interactive=True)

                kg_name = gr.Radio(list(config.kg_vector_stores.keys()),
                                label="知识库",
                                value=None,
                                info="使用知识库问答，请加载知识库",
                                interactive=True)
                set_kg_btn = gr.Button("加载知识库")

                file = gr.File(label="将文件上传到知识库，内容要尽量匹配",
                            visible=True,
                            file_types=['.txt', '.md', '.doc', '.docx', '.pdf']
                            )

            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(label='LawAssistant').style(height=400)
                with gr.Row():
                    message = gr.Textbox(label='请输入问题')
                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")
                with gr.Row():
                    gr.Markdown("""提醒：<br>
                                            LawAssistant 是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。 <br>
                                            """)
            with gr.Column(scale=2):
                search = gr.Textbox(label='搜索结果')

            # ============= 触发动作=============
            file.upload(upload_file,
                        inputs=file,
                        outputs=None)
            set_kg_btn.click(
                set_knowledge,
                show_progress='full',
                inputs=[kg_name, chatbot],
                outputs=chatbot
            )
            # 发送按钮 提交
            send.click(predict,
                    inputs=[
                        message,
                        large_language_model,
                        embedding_model,
                        top_k,
                        use_web,
                        use_pattern,
                        state
                    ],
                    outputs=[message, chatbot, state, search])

            # 清空历史对话按钮 提交
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            # 输入框 回车
            message.submit(predict,
                        inputs=[
                            message,
                            large_language_model,
                            embedding_model,
                            top_k,
                            use_web,
                            use_pattern,
                            state
                        ],
                        outputs=[message, chatbot, state, search])

    demo.queue(concurrency_count=2).launch(
        server_name='0.0.0.0',
        # server_port=8888,
        share=False,
        show_error=True,
        debug=True,
        enable_queue=True,
        inbrowser=True,
    )
