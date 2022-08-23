from flask import Flask, request
from flask_restful import Api, Resource
import json
from werkzeug.exceptions import BadRequest
import inference

app = Flask(__name__)
api = Api(app)

class QuestionAnswering(Resource):
	def get(self):
		question = request.args.get('question')
		context = request.args.get('context')
		if question == None or context == None:
			raise BadRequest('The question or the context can not be null')
		answer = inference.bertqaGetAnswers(question, context)
		return {"question": question, "context": context, "answer": answer}

api.add_resource(QuestionAnswering, "/qa")
    
if __name__ == '__main__':
    app.run(debug=True)
