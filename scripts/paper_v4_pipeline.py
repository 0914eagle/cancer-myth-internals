"""Explicit detector -> fresh response pipeline, with independently saved detections."""
import json,threading
from src.paper_baselines import DETECTORS,parse_direct,parse_review,digest,write_json
from scripts.paper_shared_extraction import SharedExtractionPipeline


class EvaluatedPipeline(SharedExtractionPipeline):
    def __init__(self,registry,call,bridge,extractor,worker):
        super().__init__(registry,call,bridge,extractor)
        self.worker=worker;self.context=threading.local()

    def run(self,method,question,source,candidate=None,gate=None):
        if method not in DETECTORS:
            return super().run(method,question,source,candidate,gate)
        candidate=candidate or {};ctx=getattr(self.context,'value',None)
        # Training evaluator calls use no test context and get no test artifact.
        path=None
        if ctx:
            row,split,qa=ctx
            name=split['name'];folder_source=source
            if qa and method in ('direct_gate','cot_gate'):
                name='fixed';folder_source='shared'
            path=self.worker.out/('qa_detections' if qa else 'table1')/folder_source/name/method/(digest([row['dataset'],row['id']])+'.json')
        if path and path.exists():
            info=json.loads(path.read_text())
            if info['question']!=question:raise ValueError('Detection question mismatch')
        else:
            info={'method':method,'question':question,'review':'','detector_raw':''}
            if ctx:info.update(id=row['id'],dataset=row['dataset'],label=None if qa else row['label'],split=name,source=folder_source)
            if method in ('tfidf_gate','probe_gate'):
                probability,threshold=gate(question)
                info.update(probability=probability,threshold=threshold,verdict=int(probability>=threshold))
            elif method=='direct_gate':
                raw=self.simple(self.prompts['direct_detector'],question)
                info.update(detector_raw=raw,verdict=parse_direct(raw))
            else:
                raw=self.simple(candidate.get('detector',self.prompts['cot_detector']),question)
                verdict,review=parse_review(raw)
                info.update(detector_raw=raw,verdict=verdict,review=review)
            info['invalid_detection']=info['verdict'] is None
            if path:write_json(path,info,immutable=True)
        if getattr(self.context,'detection_only',False):return info
        verdict=info['verdict'];review=info['review']
        prompt=self.prompts['correct_route' if verdict==1 else 'plain']
        if candidate.get('supplement'):prompt+='\n\n'+candidate['supplement']
        label='Yes' if verdict==1 else 'No' if verdict==0 else 'Invalid'
        # Both routes are newly evaluated with the actual detector output. Never
        # substitute a saved Plain or Always-correct answer based on its label.
        user=f'Question:\n{question}\n\nPremise-check result (model-generated):\nVerdict: {label}'
        if review:user+='\nReview: '+review
        if info['invalid_detection']:user+='\nRaw detector output: '+info['detector_raw']
        answer=self.simple(prompt,user)
        return {**info,'answer':answer,'raw_answer':answer,'format_invalid':False,
                'response_protocol':'fresh_response_with_saved_detection_v1'}
