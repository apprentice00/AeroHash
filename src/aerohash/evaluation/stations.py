"""Station bundles, model selection and final macro reports."""
import os
import json
import numpy as np
from loguru import logger
from .encoding import encode_array
from . import metrics as wx_eval
from aerohash.checkpoints import load_checkpoint

def load_stations(args, roots):
    use_val_select = getattr(args, 'use_val_select', True)
    eval_stations = []
    for _r in roots:
        st = {
            'name': os.path.basename(_r),
            'Xq': np.load(os.path.join(_r, 'X_query.npy')).astype(np.float32),
            'Xd': np.load(os.path.join(_r, 'X_database.npy')).astype(np.float32),
            'Sqd': np.load(os.path.join(_r, 'S_query_db.npy')),
            'qic': np.load(os.path.join(_r, 'query_is_clear.npy')),
        }
        has_val = all(os.path.isfile(os.path.join(_r, fn)) for fn in
                      ('X_val_query.npy', 'X_val_db.npy', 'S_val.npy', 'val_is_clear.npy'))
        if has_val:
            st['Xvq'] = np.load(os.path.join(_r, 'X_val_query.npy')).astype(np.float32)
            st['Xvd'] = np.load(os.path.join(_r, 'X_val_db.npy')).astype(np.float32)
            st['Sval'] = np.load(os.path.join(_r, 'S_val.npy'))
            st['vic'] = np.load(os.path.join(_r, 'val_is_clear.npy'))
        st['has_val'] = has_val
        eval_stations.append(st)
    select_on_val = use_val_select and all(s['has_val'] for s in eval_stations)
    if use_val_select and not select_on_val:
        logger.warning('[val] Missing independent validation split → falling back to test for model selection (data leakage! '
                       'Run aerohash build-val first)')
    logger.info('[select] metric={} stations={}'.format(
        'val_macro_weather_mAP' if select_on_val else 'test_macro_weather_mAP(LEAK)',
        [s['name'] for s in eval_stations]))

    return eval_stations, select_on_val


def selection_scores(model, eval_stations, select_on_val, device):
    per_overall, per_weather, per_clear, per_uniq, per_nq = [], [], [], [], []
    for st in eval_stations:
        if select_on_val:
            cq = encode_array(model, st['Xvq'], device)
            cd = encode_array(model, st['Xvd'], device)
            rep = wx_eval.stratified_report(
                wx_eval.rank_by_hamming(cq, cd), st['Sval'], st['vic'], [10, 50, 100])
        else:
            cq = encode_array(model, st['Xq'], device)
            cd = encode_array(model, st['Xd'], device)
            rep = wx_eval.stratified_report(
                wx_eval.rank_by_hamming(cq, cd), st['Sqd'], st['qic'], [10, 50, 100])
        per_overall.append(rep['overall']['mAP'] or 0.0)
        per_weather.append(rep['weather_query']['mAP'] or 0.0)
        per_clear.append(rep['clear_query']['mAP'] or 0.0)
        per_uniq.append(int(np.unique(cq, axis=0).shape[0]))
        per_nq.append(cq.shape[0])
    overall_mAP = float(np.mean(per_overall))
    weather_mAP = float(np.mean(per_weather))
    clear_mAP = float(np.mean(per_clear))
    n_uniq = int(np.sum(per_uniq))
    n_q = int(np.sum(per_nq))
    return overall_mAP, weather_mAP, clear_mAP, n_uniq, n_q


def final_evaluation(ret_path, best_mAP, eval_stations, select_on_val, device):
    ckpt_path = os.path.join(ret_path, 'model.t')
    if os.path.isfile(ckpt_path) and np.isfinite(best_mAP):
        best_model = load_checkpoint(ckpt_path, device)
        best_model.to(device)
        best_model.eval()
        per_reports, cq_all, cd_all = [], [], []
        for st in eval_stations:
            cq = encode_array(best_model, st['Xq'], device)
            cd = encode_array(best_model, st['Xd'], device)
            rep = wx_eval.stratified_report(
                wx_eval.rank_by_hamming(cq, cd), st['Sqd'], st['qic'], [10, 50, 100])
            rep['q_uniq'] = int(np.unique(cq, axis=0).shape[0])
            rep['db_uniq'] = int(np.unique(cd, axis=0).shape[0])
            rep['code_length'] = int(cq.shape[1])
            rep['station'] = st['name']
            with open(os.path.join(ret_path, 'metrics_weather_{}.json'.format(st['name'])),
                      'w', encoding='utf-8') as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            per_reports.append(rep)
            cq_all.append(cq)
            cd_all.append(cd)
            logger.info('[final_eval][{}] weather={:.5f} overall={:.5f} clear={:.5f} q_uniq={}'.format(
                st['name'], rep['weather_query'].get('mAP') or 0.0,
                rep['overall'].get('mAP') or 0.0, rep['clear_query'].get('mAP') or 0.0,
                rep['q_uniq']))

        # Aggregate macro averages (compatible with the single-station report structure for collect_multiseed)
        def _macro(group, key):
            vals = [r[group].get(key) for r in per_reports if r[group].get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None
        macro = {g: {k: _macro(g, k) for k in per_reports[0][g]
                     if k not in ('n_query', 'n_query_with_pos')}
                 for g in ('overall', 'weather_query', 'clear_query')}
        macro['stations'] = [r['station'] for r in per_reports]
        macro['per_station'] = {r['station']: r for r in per_reports}
        macro['select_metric'] = 'val_macro_weather_mAP' if select_on_val else 'test(LEAK)'
        macro['q_uniq'] = int(sum(r['q_uniq'] for r in per_reports))
        with open(os.path.join(ret_path, 'metrics_weather.json'), 'w', encoding='utf-8') as f:
            json.dump(macro, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(ret_path, 'codes_query.npy'), np.concatenate(cq_all, axis=0))
        np.save(os.path.join(ret_path, 'codes_db.npy'), np.concatenate(cd_all, axis=0))
        logger.info('[final_eval][macro] weather={} overall={} clear={} -> {}/metrics_weather.json'.format(
            macro['weather_query'].get('mAP'), macro['overall'].get('mAP'),
            macro['clear_query'].get('mAP'), ret_path))
    else:
        logger.warning('[final_eval] skip: no model.t under {}'.format(ret_path))

