"""Standalone child process; never import model weights here."""
import json
from pathlib import Path
import resource
import signal
import sys


def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    result_path=Path(sys.argv[2])
    timeout=request['timeout']
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024*1024,8*1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    task=request['task']
    if task['scorer']=='lcb':
        try:
            sys.path.insert(0,request['lcb_root'])
            from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
            outcomes,metadata=check_correctness(task['evaluation_sample'],request['code'],timeout,debug=False)
            passed=bool(len(outcomes)) and all(x == True for x in outcomes)
            result={'reward':float(passed),'success':passed,'status':'official_lcb',
                    'tests':len(outcomes),'metadata':metadata}
        except Exception as e:
            result={'status':'harness_error','error':repr(e)}
    else:
        def alarm(*_):
            raise TimeoutError('Code timed out')
        signal.signal(signal.SIGALRM,alarm)
        signal.alarm(timeout)
        try:
            scope={}
            exec(request['code'],scope)
            # MBPP fixtures may instantiate classes defined by the submitted code.
            exec(task.get('test_setup',''),scope)
            for test in task['tests']:
                exec(test,scope)
            result={'reward':1.,'success':True,'status':'all_assertions_passed','tests':len(task['tests'])}
        except BaseException as e:
            result={'reward':0.,'success':False,'status':'code_failed','error':repr(e)}
        finally:
            signal.alarm(0)
    result_path.write_text(json.dumps(result,default=str))


if __name__=='__main__':
    main()
