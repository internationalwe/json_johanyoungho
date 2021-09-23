#-*- coding:utf-8 -*-
import urllib3
import json

openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
accessKey = "07a507ca-d6b9-4a3f-849d-3e8f8c61b3d5"
analysisCode = "morp"
text = "어서오세요. 코디봇입니다. 무엇을 도와드릴까요?"

requestJson = {
	"access_key": accessKey,
	"argument": {
		"text": text,
		"analysis_code": analysisCode
	}
}

http = urllib3.PoolManager()
response = http.request(
	"POST",
	openApiURL,
	headers={"Content-Type": "application/json; charset=UTF-8"},
	body=json.dumps(requestJson)
)

print("[responseCode] " + str(response.status))
print("[responBody]")
print(str(response.data,"utf-8"))