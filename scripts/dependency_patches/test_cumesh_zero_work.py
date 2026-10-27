#!/usr/bin/env python3
"""Exercise real CuMesh connectivity kernels, including zero-edge adjacency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def fixtures():
    base=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[.5,.5,1.]]
    values={
        'single_triangle':(base,[[0,1,2]],{'face':1,'boundary':1}),
        'disjoint_triangles':(base+[[2.,0.,0.]],[[0,1,2],[3,4,5]],{'face':2,'boundary':2}),
        'connected_square':(base,[[0,1,2],[1,3,2]],{'face':1,'boundary':1}),
        'duplicate_triangle_two':(base,[[0,1,2],[0,1,2]],{'face':1,'boundary':0}),
        'duplicate_triangle_three':(base,[[0,1,2]]*3,{'face':3,'boundary':0}),
        'nonmanifold_edge':(base,[[0,1,2],[1,0,3],[0,1,4]],{'face':3,'boundary':3}),
        'empty_faces':(base,[],{'face':0}),
    }
    vertices=base[:3].copy()
    faces=[[0,1,2]]
    for vertex in range(3):
        a=len(vertices)
        xyz=vertices[vertex]
        vertices += [[xyz[0],xyz[1],xyz[2]+1.],[xyz[0]+.3,xyz[1]+.2,xyz[2]+1.4]]
        faces += [[vertex,a,a+1]]*3
    values['isolated_boundary_nonmanifold_attachments']=(vertices,faces,{'face':10,'boundary':3})
    return values


def worker(case,operation):
    import torch
    import cumesh
    vertices,faces,expected=fixtures()[case]
    v=torch.tensor(vertices,device='cuda',dtype=torch.float32)
    f=torch.tensor(faces,device='cuda',dtype=torch.int32).reshape(-1,3)
    original_v=v.clone(); original_f=f.clone()
    mesh=cumesh.CuMesh(); mesh.init(v,f)
    if operation=='face':
        mesh.get_connected_components()
        count,ids=mesh.read_connected_components()
        adjacency_count=0 if not faces else len(mesh.read_manifold_face_adjacency())
    else:
        mesh.get_boundary_connected_components()
        count,ids=mesh.read_boundary_connected_components()
        adjacency_count=len(mesh.read_manifold_boundary_adjacency())
    torch.cuda.synchronize()
    ids=ids.cpu().tolist()
    assert int(count)==expected[operation],(count,expected[operation],ids)
    assert len(set(ids))==expected[operation],ids
    expected_count=len(faces) if operation=='face' else mesh.num_boundaries
    assert len(ids)==expected_count,(len(ids),expected_count)
    if case=='isolated_boundary_nonmanifold_attachments':
        assert adjacency_count==0,adjacency_count
        if operation=='boundary':
            assert mesh.num_boundaries==3,mesh.num_boundaries
            mesh.get_boundary_loops()
            assert mesh.num_boundary_loops==0,mesh.num_boundary_loops
            # Even a generous fill threshold must not join isolated boundaries.
            mesh.fill_holes(max_hole_perimeter=100.)
    out_v,out_f=mesh.read()
    assert torch.equal(out_v,original_v) and torch.equal(out_f,original_f),'Connectivity changed input geometry'
    return {'status':'pass','components':int(count),'ids':ids,'manifold_adjacency_count':adjacency_count,'input_faces_preserved':len(faces),'input_vertices_preserved':len(vertices)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=fixtures())
    p.add_argument('--operation',choices=['face','boundary'])
    p.add_argument('--phase',choices=['before','after'],default='after')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.case:
        try:
            result=worker(args.case,args.operation)
        except Exception as error:
            result={'status':'error','exception':repr(error)}
        print(json.dumps(result),flush=True)
        return 0 if result['status']=='pass' else 2
    checks=[]
    before_failures={('single_triangle','face'),('disjoint_triangles','face'),('duplicate_triangle_three','face'),('nonmanifold_edge','face'),('isolated_boundary_nonmanifold_attachments','face'),('isolated_boundary_nonmanifold_attachments','boundary'),('empty_faces','face')}
    for case,(_,_,operations) in fixtures().items():
        for operation in operations:
            child=subprocess.run([sys.executable,__file__,'--case',case,'--operation',operation],capture_output=True,text=True,env={**os.environ,'CUDA_LAUNCH_BLOCKING':'1'})
            try: result=json.loads(child.stdout.strip().splitlines()[-1])
            except Exception: result={'status':'error','stdout':child.stdout[-3000:],'stderr':child.stderr[-3000:]}
            result.update({'case':case,'operation':operation,'returncode':child.returncode})
            expected_error=args.phase=='before' and (case,operation) in before_failures
            result['expected_before_error']=expected_error
            result['expectation_met']=(result['status']=='error' and 'invalid configuration' in result.get('exception','')) if expected_error else result['status']=='pass'
            checks.append(result)
            print(case,operation,result['status'],'expected_error='+str(expected_error),flush=True)
    report={'phase':args.phase,'status':'pass' if all(x['expectation_met'] for x in checks) else 'fail','checks':checks}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'output':str(args.output)}),flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__': raise SystemExit(main())
