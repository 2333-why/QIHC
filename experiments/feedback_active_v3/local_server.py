"""Loopback adapter for the historical Transformers LocalModel backend."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def generate(generator, body):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from qihc.modeling.generation import complete_object
    prompt=body['messages'][0]['content']
    inputs=generator._inputs(prompt)
    inputs={k:v.to(generator.device) for k,v in inputs.items()}
    width=inputs['input_ids'].shape[1]; budget=int(body['max_tokens'])
    if width+budget>generator.max_input_tokens: raise ValueError('Context limit exceeded; no truncation')
    class Complete(StoppingCriteria):
        def __call__(self, ids, scores, **kwargs):
            return complete_object(generator.tokenizer.decode(ids[0,width:],skip_special_tokens=True)) is not None
    torch.manual_seed(int(body['seed']))
    temperature=float(body.get('temperature',0))
    kwargs=dict(max_new_tokens=budget,do_sample=temperature>0,
                pad_token_id=generator.tokenizer.pad_token_id or generator.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Complete()]),max_time=150)
    if temperature>0: kwargs.update(temperature=temperature,top_p=1.0,top_k=0)
    with torch.inference_mode(): output=generator.model.generate(**inputs,**kwargs)
    tokens=output[0,width:]; text=generator.tokenizer.decode(tokens,skip_special_tokens=True)
    complete=complete_object(text)
    return dict(choices=[dict(finish_reason='stop' if complete or len(tokens)<budget else 'length',
                             message=dict(role='assistant',content=complete or text))],
                usage=dict(prompt_tokens=width,completion_tokens=len(tokens)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--max-model-len',type=int,default=32768);a=p.parse_args()
    from qihc.modeling.backend import LocalModel
    generator=LocalModel(a.model,device='cuda:0',max_input_tokens=a.max_model_len)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self): self.reply(200 if self.path=='/health' else 404,{})
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.reply(404,{});return
            try:
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.reply(200,generate(generator,body))
            except Exception as exc:
                import traceback
                traceback.print_exc();self.reply(500,dict(error=str(exc)))
    print('Historical Transformers LocalModel ready',flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
