content = open('src/db.py', 'r', encoding='utf-8').read()
print("save_detection 포함 여부:", 'save_detection' in content)
print("파일 글자 수:", len(content))