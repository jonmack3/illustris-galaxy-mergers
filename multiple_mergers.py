# -*- coding: utf-8 -*-
"""
Obtain/output/plot Illustris multiple galaxy merger data.
Created on Thu ‎Jun ‎23 16:20:59 ‎2016
@author: Jonathan Mack
"""

import logging
import math
import os
import platform
import sys
import time
import warnings

import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import config as cfg
import globals as glb
import illustris_python as il

# np.set_printoptions(threshold=sys.maxsize)

def main(argv):
    """
    Execute the main function.

    Parameters
    ----
    argv : int
        If length zero, use parameters in config.py. If length eight,
        use those. If any other, raise an execption.
    """
    timestart = time.time()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if cfg.debug == 1:
        loglvl = logging.DEBUG
    else:
        loglvl = logging.WARNING
    logging.basicConfig(filename='multiple_mergers.log', filemode='w',
                        level=loglvl)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.ticker').disabled = True

    # set run parameters
    if len(argv) == 0:
        fnum = cfg.fnum
        ilnum = cfg.ilnum
        snapnum = cfg.snapnum
        mu_max = cfg.mu_max
        virtualprog = cfg.virtualprog
        SubLink_gal = cfg.SubLink_gal
        subhalostart = cfg.subhalostart
        subhalo_end = cfg.subhalo_end
    elif len(argv) == 8:
        fnum = int(argv[0])
        ilnum = int(argv[1])
        snapnum = int(argv[2])
        mu_max = int(argv[3])
        virtualprog = int(argv[4])
        SubLink_gal = int(argv[5])
        subhalostart = int(argv[6])
        subhalo_end = int(argv[7])
    else:
        raise ValueError('0 (default) or 8 arguments required: fnum, '
                         'ilnum, snapnum, mu_max, virtualprog, '
                         'SubLink_gal, subhalostart, and subhalo_end')
    if (not isinstance(fnum, int) or not isinstance(ilnum, int)
        or not isinstance(snapnum, int) or not isinstance(mu_max, int)
        or not isinstance(virtualprog, int) or not isinstance(SubLink_gal, int)
        or not isinstance(subhalostart, int)
        or not isinstance(subhalo_end, int)):
            raise ValueError('fnum, ilnum, snapnum, mu_max, SubLink_gal, '
                             'subhalostart, and subhalo_end must all be int')
    if ilnum not in [1, 3, 100, 300]:
        raise ValueError('Invalid ilnum')
    if fnum < 0 or fnum > len(cfg.functions):
        raise ValueError('Invalid function number')
    if virtualprog != 0 and virtualprog != 1:
        raise ValueError('Invalid virtualprog value')
    if SubLink_gal != 0 and SubLink_gal != 1:
        raise ValueError('Invalid SubLink_gal value')

    mu_min = 1 / mu_max

    if cfg.functions[fnum] == 'setmergers':
        setmergers(ilnum, snapnum, mu_min, mu_max, virtualprog,
                   SubLink_gal, subhalostart, subhalo_end)
    elif cfg.functions[fnum] == 'setmergersmult':
        setmergersmult()
    elif cfg.functions[fnum] == 'viewmergerdata':
        viewmergerdata()
    elif cfg.functions[fnum] == 'setmassmax':
        setmassmax()
    elif cfg.functions[fnum] == 'setfs':
        setfs()
    elif cfg.functions[fnum] == 'viewfvmdata':
        viewfvmdata()
    elif cfg.functions[fnum] == 'setdtmax':
        setdtmax()
    elif cfg.functions[fnum] == 'setdts':
        setdts()
    elif cfg.functions[fnum] == 'viewdtdata':
        viewdtdata()
    elif cfg.functions[fnum] == 'setfmin':
        setfmin()
    elif cfg.functions[fnum] == 'createfvmplots':
        createfvmplots()
    elif cfg.functions[fnum] == 'createfvmratioplots':
        createfvmratioplots()
    elif cfg.functions[fnum] == 'create_dtplots':
        create_dtplots()
    elif cfg.functions[fnum] == 'createpubplots':
            createpubplots()
    elif cfg.functions[fnum] == 'test':
        test()
    logging.shutdown()
    time_end = time.time()
    exectime = time_end - timestart
    print(f'\nExecution time {exectime // 60:.0f} min {exectime % 60:.2f} s')

def get_run_info(ilnum):
    """
    Get dictionary number, base path, directory name, maximum snapnum,
    dimensionless Hubble parameter, and number of subgroups from Illustris run
    number.

    Parameters
    ----
    ilnum : int
        Illustris run number.

    Returns
    ----
    tuple : (int, int, int, int, float, int)
        Tuple of dictionary number, basepath, directory name, snapnummax, h, 
        and numsubgroups.
    """
    # get dictionary number, directory name, max snapnum, and h
    if ilnum == 1 or ilnum == 3:
        dictnum = 0
        ilrun = 'Il-' + str(ilnum)
        snapnummax = 135
        h = glb.hOG
    elif ilnum == 100 or ilnum == 300:
        dictnum = 1
        ilrun = 'TNG' + str(ilnum)
        snapnummax = 99
        h = glb.hTNG
        
    # get basepath and number of subgroups
    if ilnum == 1:
        basepath = os.path.join('SimulationData', 'L75n1820FP')
        numsubgroups = 4366546
    elif ilnum == 3:
        basepath = os.path.join('SimulationData', 'L75n455FP')
        numsubgroups = 121209
    elif ilnum == 100:
        basepath = os.path.join('SimulationData', 'L75n1820TNG')
        numsubgroups = 4371211
    elif ilnum == 300:
        basepath = os.path.join('SimulationData', 'L205n2500TNG')
        numsubgroups = 14485709
            
    return (dictnum, basepath, ilrun, snapnummax, h, numsubgroups)

def setmergers(ilnum, snapnum, mu_min, mu_max, virtualprog, SubLink_gal,
               subhalostart, subhalo_end):
    """
    Get relevant mergers from the Illustris merger trees.

    Parameters
    ----
    ilnum : int
        Illustris run number.
    snapnum : int
        Starting analysis snapnum.
    mu_min : float
        Minimum acceptable valid merger mass ratio.
    mu_max : float
        Maximum acceptable valid merger mass ratio.
    SubLink_gal : boolean
        SubLink trees used if 0, SubLink_gal if 1
    subhalostart : int
        SubfindID of root of beginning merger tree.
    subhalo_end : int
        SubfindID of root of ending merger tree, -1 if to be last tree.
    """
    print(f'Running setmergers ilnum {ilnum} snapnum {snapnum} mu_min '
          f'{mu_min} mu_max {mu_max} virtualprog {virtualprog} '
          f'SubLink_gal {SubLink_gal} subhalostart {subhalostart} '
          f'subhalo_end {subhalo_end}')
    fields = ['SubhaloID', 'SubhaloIDRaw', 'FirstProgenitorID',
              'NextProgenitorID', 'DescendantID', 'SubhaloMassInRadType']
    dtype = [('s_id', np.int64), ('m', np.float_)]
    subhalos = np.zeros(cfg.setmergers_arylen, dtype=dtype)
    dtype = [('s_id_a', np.int64), ('s_id_m', np.int64), ('snap_m', np.int16),
             ('m_pri', np.float_), ('m_sec', np.float_), ('m_dsc', np.float_)]
    mergers = np.zeros(cfg.setmergers_arylen, dtype=dtype)
    mrgrs_same_t = np.zeros(cfg.setmergers_arylen, dtype=np.float_)
    no_tree = np.zeros(cfg.setmergers_arylen, dtype=np.int64)
    i_mgr, i_sh, i_st, i_nt = 0, 0, 0, 0

    # set run-specific info
    dictnum, basepath, ilrun, snapnummax, h, numsubgroups = get_run_info(ilnum)
    if subhalo_end == -1:
        subhalo_endnum = numsubgroups
    else:
        subhalo_endnum = subhalo_end
    ta = glb.ts[dictnum][snapnum]
    Ta = glb.Tsnys[dictnum][snapnum]
    Tsnp = glb.Tsnps[dictnum][snapnum]
    if SubLink_gal == 0:
        treename = 'SubLink'
    else:
        treename = 'SubLink_gal'
    logging.debug(f'dictnum {dictnum} h {h} ta {ta} Ta {Ta} Tsnp {Tsnp} '
                  f'treename {treename}')

    # get masses of all tree roots

    mhs_by_s_id = il.groupcat.loadSubhalos(
            os.path.join(basepath, 'output'), snapnummax,
            fields=['SubhaloMassInRadType'])[:,4]
    ms_by_s_id = mhs_by_s_id / h

    # analyze each tree
    for sh_num in range(subhalostart, subhalo_endnum):
        print(f'\rAnalyzing tree at sh #: {sh_num}', end='')
        logging.debug(f'tree at sh_num {sh_num}')

        # skip trees whose root is less than mmin
        if ms_by_s_id[sh_num] < cfg.mmin:
            logging.debug(f'ms_by_s_id[sh_num] {ms_by_s_id[sh_num]}: '
                          'root < mmin; continue')
            continue

        try:
            tree = il.sublink_j.loadTree(basepath, snapnummax, sh_num,
                                         fields=fields, treeName=treename)
                    
            # check each subhalo in the current tree
            for i in range(tree['SubhaloIDRaw'].size):
                # analyze only subhalos at correct snapshot and minimum mass
                m_mult_h = tree['SubhaloMassInRadType'][:, 4][i]
                m_sh = m_mult_h / h
                logging.debug(f"i {i} IDraw {tree['SubhaloIDRaw'][i]} "
                              f"ID {tree['SubhaloID'][i]} "
                              f"fpID {tree['FirstProgenitorID'][i]} "
                              f"npID {tree['NextProgenitorID'][i]} "
                              f"dID {tree['DescendantID'][i]} "
                              f'm*h {m_mult_h} m {m_sh}')

                # subhalo too small or at wrong snap; move on to next one
                if (int(tree['SubhaloIDRaw'][i]//1e12) != snapnum
                        or m_sh < cfg.mmin):
                    continue
                subhalos[i_sh] = (tree['SubhaloID'][i], m_sh)
                i_sh += 1
                logging.debug(f'subhalo valid: subhalos\n{subhalos}')

                # get all same_t mergers: begin
                logging.debug('same_t mergers')
                j = i

                if tree['FirstProgenitorID'][j] != -1:
                    # get progenitor and any siblings
                    j = j + (tree['FirstProgenitorID'][j] -
                             tree['SubhaloID'][j])
                    sibs = [j]
                    while tree['NextProgenitorID'][j] != -1:
                        j = j + (tree['NextProgenitorID'][j]
                                 - tree['SubhaloID'][j])
                        sibs.append(j)
                    logging.debug(f'sibs {sibs}')

                    if len(sibs) > 1:
                        m_mpm, snap_mpm = -1, -1
                        for k in range(1, len(sibs)):
                            logging.debug(f'k {k} sibs[k] {sibs[k]}')

                            # get sibling max past mass
                            mhmax = -1
                            endofbranch = False
                            n = sibs[k]
                            while not endofbranch:
                                mh_inst = tree['SubhaloMassInRadType'][:, 4][n]
                                if mh_inst > mhmax:
                                    mhmax = mh_inst
                                    snap_mpm = int(
                                            tree['SubhaloIDRaw'][n]//1e12)
                                logging.debug(
                                        f'ind_inst {n} mh_inst {mh_inst} '
                                        f'mhmax {mhmax} snap_mpm {snap_mpm} ')
                                if tree['FirstProgenitorID'][n] != -1:
                                    n = n + (tree['FirstProgenitorID'][n]
                                             - tree['SubhaloID'][n])
                                else:
                                    endofbranch = True
                            m_mpm = mhmax / h
                            logging.debug(f'mhmax {mhmax} m_mpm {m_mpm}')

                            # get mass of primary progenitor
                            n = sibs[0]
                            while ((int(tree['SubhaloIDRaw'][n]//1e12) >
                                    snap_mpm) and
                                    tree['FirstProgenitorID'][n] != -1):
                                n = n + (tree['FirstProgenitorID'][n]
                                         - tree['SubhaloID'][n])
                            mh_pri = tree['SubhaloMassInRadType'][:, 4][n]
                            m_pri = mh_pri / h
                            logging.debug(f'mh_pri {mh_pri} m_pri {m_pri}')

                            # add mass of analysis subhalo if merger valid
                            if m_pri > 0:
                                ratio = m_mpm / m_pri
                                logging.debug(f'ratio {ratio}')

                                if ratio >= mu_min and ratio <= mu_max:
                                    mrgrs_same_t[i_st] = m_sh
                                    i_st += 1
                                    logging.debug('ratio valid')
                logging.debug(f'mrgrs_same_t\n{mrgrs_same_t}')

                # get previous mergers: begin
                logging.debug('prev mergers')
                test_indcs, dsc_snaps = [], []
                num_cant_ovrlp, num_must_ovrlp = 0, 0
                j = i
                stop = False

                # since snapnum of merger defined as post-merge value, must go
                # to first progenitor's snapshot before starting search
                if tree['FirstProgenitorID'][j] == -1:
                    stop = True
                    logging.debug('no s_a fp: start next merger analysis')
                else:
                    j = j + (tree['FirstProgenitorID'][j] -
                             tree['SubhaloID'][j])
                    dsc_snap = snapnum
                    test_indcs.append(j)
                    dsc_snaps.append(dsc_snap)
                    logging.debug(f'init: test_indcs {test_indcs}\n'
                                  f'dsc_snaps {dsc_snaps}')

                # move up tree, analyzing mergers
                while (len(test_indcs) != 0 and not stop):
                    # dsc_snap not in dsc_snaps; decrement it
                    if dsc_snap not in dsc_snaps:
                        dsc_snap -= 1
                        logging.debug('dsc_snap not in dsc_snaps; '
                                      f'decremented to {dsc_snap}')
                        continue

                    # get any siblings
                    j = test_indcs[dsc_snaps.index(dsc_snap)]
                    sibs = [j]
                    sib = j
                    while tree['NextProgenitorID'][sib] != -1:
                        sib = sib + (tree['NextProgenitorID'][sib]
                                     - tree['SubhaloID'][sib])
                        sibs.append(sib)
                    logging.debug(f'j {j} sibs {sibs}')

                    if len(sibs) > 1:
                        # get age-related data
                        tms = glb.ts[dictnum][dsc_snap-1]
                        tme = glb.ts[dictnum][dsc_snap]
                        Tm = glb.Tsnys[dictnum][dsc_snap]
                        dsc = j + (tree['DescendantID'][j]
                                   - tree['SubhaloID'][j])
                        m_dsc = tree['SubhaloMassInRadType'][:, 4][dsc] / h
                        logging.debug(f'>1 sib: tms {tms} tme {tme} Tm {Tm} '
                                      f'dsc {dsc} m_dsc {m_dsc}')

                        # get sibling max past masses
                        m_mpms, snap_mpms = [-1] * len(sibs), [-1] * len(sibs)
                        for k in range(len(sibs)):
                            mhmax = -1
                            endofbranch = False
                            n = sibs[k]
                            while not endofbranch:
                                mh_inst = tree['SubhaloMassInRadType'][:, 4][n]
                                if mh_inst > mhmax:
                                    mhmax = mh_inst
                                    snap_mpms[k] = int(
                                            tree['SubhaloIDRaw'][n]//1e12)
                                logging.debug(
                                        f'ind_inst {n} mh_inst {mh_inst} '
                                        f'mhmax {mhmax} snap_mpms {snap_mpms}')
                                if tree['FirstProgenitorID'][n] != -1:
                                    n = n + (tree['FirstProgenitorID'][n]
                                             - tree['SubhaloID'][n])
                                else:
                                    endofbranch = True
                            m_mpms[k] = mhmax / h

                        # adjust m_dsc/create virtual progenitor as needed
                        summ_mpms = sum(m_mpms)
                        logging.debug(f'before virtual sh creation: '
                                      f'sibs {sibs} snap_mpms {snap_mpms} '
                                      f'm_mpms {m_mpms} summ_mpms {summ_mpms} '
                                      f'm_dsc {m_dsc}')
                        if summ_mpms > m_dsc:
                            logging.debug('adjust m_dsc')
                            m_dsc = summ_mpms
                        elif (virtualprog == 1 and m_dsc > 0
                              and summ_mpms / m_dsc < cfg.mminvirt):
                            m_vrtl = m_dsc - summ_mpms
                            if m_vrtl > m_mpms[0]:
                                logging.debug('create virtual fp')
                                sibs.insert(0, -1)
                                m_mpms.insert(0, m_vrtl)
                                snap_mpms.insert(0, -1)
                            else:
                                logging.debug('create virtual sp')
                                sibs.append(-1)
                                m_mpms.append(m_vrtl)
                                snap_mpms.append(-1)
                        logging.debug(f'after creation: sibs {sibs} '
                                      f'snap_mpms {snap_mpms} m_mpms {m_mpms} '
                                      f'summ_mpms {sum(m_mpms)} m_dsc {m_dsc}')

                        # check for valid mergers
                        for k in range(1, len(sibs)):
                            logging.debug(f'sibs[0] {sibs[0]} k {k} '
                                          f'sibs[k] {sibs[k]}')

                            # get mass of primary progenitor
                            if sibs[0] == -1 or sibs[k] == -1:
                                m_pri = m_mpms[0]
                                logging.debug(f'pp or sp = -1, m_pri {m_pri}')
                            else:
                                n = sibs[0]
                                while ((int(tree['SubhaloIDRaw'][n]//1e12) >
                                        snap_mpms[k]) and
                                        tree['FirstProgenitorID'][n] != -1):
                                    n = n + (tree['FirstProgenitorID'][n]
                                             - tree['SubhaloID'][n])
                                mh_pri = tree['SubhaloMassInRadType'][:, 4][n]
                                m_pri = mh_pri / h
                                logging.debug(f'mh_pri {mh_pri} m_pri {m_pri}')

                            # m_pri = 0; try next sibling pair
                            if m_pri == 0:
                                logging.debug('m_pri = 0; try next')
                                break

                            # get ratio
                            m_sec = m_mpms[k]
                            ratio = m_sec / m_pri
                            logging.debug(f'ratio {ratio}')

                            if ratio < mu_min or ratio > mu_max:
                                logging.debug('ratio high/low; try next')
                                continue

                            # has valid ratio: add to results
                            mergers[i_mgr] = ((tree['SubhaloID'][i],
                                               tree['SubhaloID'][dsc],
                                               dsc_snap, m_pri, m_sec, m_dsc))
                            i_mgr += 1
                            logging.debug(f'ratio valid: mergers\n{mergers}')

                            # set Tmin, Tmax
                            Tmin = min(Ta, Tm, Tsnp) * min(cfg.Tfacs)
                            Tmax = max(Ta, Tm, Tsnp) * max(cfg.Tfacs)
                            logging.debug(f'Tmin {Tmin} Tmax {Tmax} tms {tms} '
                                          f'tme {tme} ta {ta}')

                            # increment "can't overlap" counter, if needed                            
                            if tme + Tmax / 2 < ta:
                                num_cant_ovrlp += 1
                                logging.debug(
                                        f"can't++, can't {num_cant_ovrlp}")
                                
                                # can't-overlap ctr equals 2: exit search
                                if num_cant_ovrlp == 2:
                                    logging.debug("break: can't overlap == 2")
                                    stop = True
                                    break

                            # increment "must overlap" counter, if needed
                            elif tms + Tmin / 2 >= ta:
                                num_must_ovrlp += 1
                                logging.debug(f'must++, must {num_must_ovrlp}')
                            
                                # must-overlap ctr equals 5: exit search
                                if num_must_ovrlp == 5:
                                    logging.debug("break: must overlap == 5")
                                    stop = True
                                    break

                            # add sec sibling to list of indices to check
                            if (sibs[k] != -1 and
                                    tree['FirstProgenitorID'][sibs[k]] != -1):
                                fpsec = sibs[k] + (
                                        tree['FirstProgenitorID'][sibs[k]]
                                        - tree['SubhaloID'][sibs[k]])
                                test_indcs.append(fpsec)
                                dsc_snaps.append(int(tree['SubhaloIDRaw']
                                                     [sibs[k]]//1e12))
                                logging.debug('sec fp added: '
                                              f'test_indcs {test_indcs} '
                                              f'dsc_snaps {dsc_snaps}')

                    # add pri sibling to list of indices to check
                    if (not stop and sibs[0] != -1
                            and tree['FirstProgenitorID'][j] != -1):
                        fp_pri = j + (tree['FirstProgenitorID'][j]
                                      - tree['SubhaloID'][j])
                        test_indcs.append(fp_pri)
                        dsc_snaps.append(int(tree['SubhaloIDRaw'][j]//1e12))
                        logging.debug(f'pri fp added: test_indcs {test_indcs} '
                                      f'dsc_snaps {dsc_snaps}')

                    # remove tested index
                    if not stop:
                        pop_ind = test_indcs.index(j)
                        test_indcs.pop(pop_ind)
                        dsc_snaps.pop(pop_ind)
                        logging.debug(f'pop curr: test_indcs {test_indcs} '
                                      f'dsc_snaps {dsc_snaps}')

                # get next mergers: begin
                logging.debug('next mergers')
                num_cant_ovrlp, num_must_ovrlp = 0, 0
                j = i
                stop = False

                # move down tree until stopping condition reached
                while (not stop):
                    # check for tree root reached
                    if tree['DescendantID'][j] == -1:
                        logging.debug('tree root reached; exit search')
                        break
                    else:
                        dsc = j + (tree['DescendantID'][j]
                                   - tree['SubhaloID'][j])

                    # get any siblings
                    fp = dsc + (tree['FirstProgenitorID'][dsc]
                                - tree['SubhaloID'][dsc])
                    sibs = [fp]
                    sib = fp
                    while tree['NextProgenitorID'][sib] != -1:
                        sib = sib + (tree['NextProgenitorID'][sib]
                                     - tree['SubhaloID'][sib])
                        sibs.append(sib)
                    logging.debug(f'sibs {sibs}')

                    if len(sibs) == 1:
                        logging.debug('no siblings; try descendant')
                        j = dsc
                        continue

                    # get age-related data
                    dsc_snap = int(tree['SubhaloIDRaw'][dsc]//1e12)
                    tms = glb.ts[dictnum][dsc_snap-1]
                    tme = glb.ts[dictnum][dsc_snap]
                    Tm = glb.Tsnys[dictnum][dsc_snap]
                    m_dsc = tree['SubhaloMassInRadType'][:, 4][dsc] / h
                    logging.debug(f'j {j} dsc_snap {dsc_snap} tms {tms} '
                                  f'tme {tme} Tm {Tm} m_dsc {m_dsc}')

                    # get sibling max past masses
                    m_mpms, snap_mpms = [-1] * len(sibs), [-1] * len(sibs)
                    for k in range(len(sibs)):
                        mhmax = -1
                        endofbranch = False
                        n = sibs[k]
                        while not endofbranch:
                            mh_inst = tree['SubhaloMassInRadType'][:, 4][n]
                            if mh_inst > mhmax:
                                mhmax = mh_inst
                                snap_mpms[k] = int(
                                        tree['SubhaloIDRaw'][n]//1e12)
                            logging.debug(
                                    f'ind_inst {n} mh_inst {mh_inst} '
                                    f'mhmax {mhmax} snap_mpms {snap_mpms}')
                            if tree['FirstProgenitorID'][n] != -1:
                                n = n + (tree['FirstProgenitorID'][n]
                                         - tree['SubhaloID'][n])
                            else:
                                endofbranch = True
                        m_mpms[k] = mhmax / h

                    # adjust m_dsc/create virtual progenitor as needed
                    summ_mpms = sum(m_mpms)
                    logging.debug(f'before virtual sh creation: sibs {sibs} '
                                  f'snap_mpms {snap_mpms} m_mpms {m_mpms} '
                                  f'summ_mpms {summ_mpms} m_dsc {m_dsc}')
                    if summ_mpms > m_dsc:
                        logging.debug('adjust m_dsc')
                        m_dsc = summ_mpms
                    elif (virtualprog == 1 and m_dsc > 0
                          and summ_mpms / m_dsc < cfg.mminvirt):
                        m_vrtl = m_dsc - summ_mpms
                        if m_vrtl > m_mpms[0]:
                            logging.debug('create virtual fp')
                            sibs.insert(0, -1)
                            m_mpms.insert(0, m_vrtl)
                            snap_mpms.insert(0, -1)
                        else:
                            logging.debug('create virtual sp')
                            sibs.append(-1)
                            m_mpms.append(m_vrtl)
                            snap_mpms.append(-1)
                    logging.debug(f'after creation: sibs {sibs} '
                                  f'snap_mpms {snap_mpms} m_mpms {m_mpms} '
                                  f'summ_mpms {sum(m_mpms)} m_dsc {m_dsc}')

                    # check siblings
                    for k in range(len(sibs)):
                        logging.debug(f'k {k} sibs[k] {sibs[k]}')

                        # current and test indices equal, try next pair
                        if sibs[k] == j:
                            logging.debug('current = test; try next pair')
                            continue

                        # if current not fp, test only with fp
                        if k != 0 and j != sibs[0]:
                            logging.debug('either current or test must be fp')
                            continue

                        # set primary and secondary progenitors
                        if k > sibs.index(j):
                            ind_pri = j
                            ind_sec = sibs[k]
                        else:
                            ind_pri = sibs[k]
                            ind_sec = j
                            logging.debug('k < sibs.index(j)')
                        logging.debug(f'ind_pri {ind_pri} ind_sec {ind_sec}')

                        # get mass of primary progenitor
                        if ind_pri == -1 or ind_sec == -1:
                            m_pri = m_mpms[sibs.index(ind_pri)]
                            logging.debug(f'pp or sp = -1, m_pri {m_pri}')
                        else:
                            n = ind_pri
                            while ((int(tree['SubhaloIDRaw'][n]//1e12)
                                    > snap_mpms[sibs.index(ind_sec)])
                                    and tree['FirstProgenitorID'][n] != -1):
                                n = n + (tree['FirstProgenitorID'][n]
                                         - tree['SubhaloID'][n])
                            mh_pri = tree['SubhaloMassInRadType'][:, 4][n]
                            m_pri = mh_pri / h
                            logging.debug(f'mh_pri {mh_pri} m_pri {m_pri}')

                        # m_pri = 0; try next sibling pair
                        if m_pri == 0:
                            logging.debug('m_pri = 0; try next')
                            continue

                        # get ratio
                        m_sec = m_mpms[sibs.index(ind_sec)]
                        ratio = m_sec / m_pri
                        logging.debug(
                                f'm_sec {m_sec} m_pri {m_pri} ratio {ratio}')

                        # test for non-valid ratio exit conditions
                        logging.debug(
                                f'ratio {ratio} sibs_ind {sibs.index(j)}')
                        if ratio < mu_min and sibs.index(j) != 0:
                            logging.debug('subsumed; end search')
                            stop = True
                            break
                        if ratio < mu_min or ratio > mu_max:
                            logging.debug('ratio high/low; trying next pair')
                            continue

                        # has valid ratio: add to results
                        mergers[i_mgr] = ((tree['SubhaloID'][i],
                                           tree['SubhaloID'][dsc],
                                           dsc_snap, m_pri, m_sec, m_dsc))
                        i_mgr += 1
                        logging.debug(f'ratio valid: mergers\n{mergers}')

                        # set Tmin, Tmax
                        Tmin = min(Ta, Tm, Tsnp) * min(cfg.Tfacs)
                        Tmax = max(Ta, Tm, Tsnp) * max(cfg.Tfacs)
                        logging.debug(f'Tmin {Tmin} Tmax {Tmax} tms {tms} '
                                      f'tme {tme} ta {ta}')

                        # increment "can't overlap" counter, if needed
                        if tms - Tmax / 2 > ta:
                            num_cant_ovrlp += 1
                            logging.debug(f"can't++, can't {num_cant_ovrlp}")
                            
                            # can't-overlap ctr equals 2: exit search
                            if num_cant_ovrlp == 2:
                                logging.debug("break: can't overlap == 2")
                                stop = True
                                break

                        # increment "must overlap" counter, if needed
                        elif tme - Tmin / 2 <= ta:
                            num_must_ovrlp += 1
                            logging.debug(f'must++, must {num_must_ovrlp}')
                            
                            # must-overlap ctr equals 5: exit search
                            if num_must_ovrlp == 5:
                                logging.debug("break: must overlap == 5")
                                stop = True
                                break

                    # no exit conditions satisfied; move to descendant
                    j = dsc

                # subhalo analyzed; show result
                logging.debug(f'subhalo analyzed: mergers\n{mergers}')

        # handle tree-not-found errors
        except TypeError:
            no_tree[i_nt] = sh_num
            i_nt += 1

    # analysis complete; eliminate unused array rows
    logging.debug(f'analysis complete\nsubhalos\n{subhalos}\nmergers\n'
                  f'{mergers}\nmrgrs_same_t\n{mrgrs_same_t}\n'
                  f'no_tree\n{no_tree}')
    subhalos = np.resize(subhalos, i_sh)
    mergers = np.resize(mergers, i_mgr)
    mrgrs_same_t = np.resize(mrgrs_same_t, i_st)
    no_tree = np.resize(no_tree, i_nt)
    logging.debug(f'after resize\nsubhalos\n{subhalos}\nmrgrs_same_t\n'
                  f'{mrgrs_same_t}\nmergers\n{mergers}\nno_tree\n{no_tree}')

    # write arrays to file
    pathname = os.path.join('output', 'numerical', 'mrgr', ilrun)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    f_cfg_mgr = (f'i{ilnum}s{snapnum}rl{1/mu_max:1.2f}'
                 f'ru{mu_max:02d}v{virtualprog}g{SubLink_gal}'
                 f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{subhalostart}se{subhalo_end}')
    np.savez_compressed(os.path.join(pathname, 'mrgrdat' + f_cfg_mgr + '.npz'),
                        subhalos=subhalos, mergers=mergers,
                        mrgrs_same_t=mrgrs_same_t, no_tree=no_tree)

def setmergersmult():
    """
    Conduct multiple setmergers runs.
    """
    print('Running setmergersmult')

    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG

        for j, k, n, o in it.product(snapnums, cfg.mu_maxes, cfg.virtualprogs,
                                     cfg.SubLink_gals):
            setmergers(ilnum, j, 1/k, k, n, o, cfg.subhalostart,
                       cfg.subhalo_end)

def viewmergerdata():
    """
    View a setmergers run's subhalos, mergers, and those with no tree.
    """
    print(f'Running viewmergerdata ilnum {cfg.ilnum} snapnum {cfg.snapnum} '
          f'mu_min {1/cfg.mu_max} mu_max {cfg.mu_max} '
          f'virtualprogs {cfg.virtualprog} SubLink_gal {cfg.SubLink_gal} '
          f'mmin {cfg.mmin:1.1f} mminvirt {cfg.mminvirt:1.2f} '
          f'subhalostart {cfg.subhalostart} subhalo_end {cfg.subhalo_end}')

    _, _, ilrun, _, _, _ = get_run_info(cfg.ilnum)

    f_cfg_mgr = (f'i{cfg.ilnum}s{cfg.snapnum}rl{1/cfg.mu_max:1.2f}'
                 f'ru{cfg.mu_max:02d}v{cfg.virtualprog}'
                 f'g{cfg.SubLink_gal}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with np.load(os.path.join('output', 'numerical', 'mrgr', ilrun,
                              'mrgrdat' + f_cfg_mgr + '.npz')) as data:
        with open('mrgrdat' + f_cfg_mgr + '.txt', 'w') as f:
            for i in data:
                f.write(i + '\n')
                for j in range(len(data[i])):
                    f.write(str(data[i][j]) + '\n')

def setmassmax():
    """
    Set maximum subhalo mass across a range of specified parameters.
    """
    print(f'Running setmassmax subhalostart {cfg.subhalostart} '
          f'subhalo_end {cfg.subhalo_end}')

    # find mass max
    massmax = 0
    ils = ''.join(map(str, cfg.ilnums))
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        _, basepath, ilrun, snapnummax, _, _ = get_run_info(ilnum)

        for j, k, n, o in it.product(snapnums, cfg.mu_maxes, cfg.virtualprogs,
                                     cfg.SubLink_gals):
            logging.debug(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                          f'virtualprogs {n} SubLink_gals {o}')
            print(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                  f'virtualprogs {n} SubLink_gals {o}', end=' ')

            f_cfg_mgr = (f'i{ilnum}s{j}rl{1/k:1.2f}ru{k:02d}v{n}g{o}'
                         f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                         f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'mrgr', ilrun,
                                      'mrgrdat' + f_cfg_mgr + '.npz')) as data:
                subhalos = data['subhalos']
            logging.debug(f'subhalos\n{subhalos}')
            massmaxtemp = np.amax(subhalos['m'])
            if massmaxtemp > massmax:
                massmax = massmaxtemp
            logging.debug(f'massmaxtemp {massmaxtemp} massmax {massmax}')
            print(f'massmaxtemp {massmaxtemp} massmax {massmax}')

    # display/store mass max
    logging.debug(f'Mass max {massmax}')
    print(f'Mass max {massmax}')
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    pathname = os.path.join('output', 'numerical', 'mmax')
    if (not os.path.exists(pathname)):
        os.makedirs(pathname)
    fcfg_mmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join(pathname, 'massmax' + fcfg_mmax + '.txt'), 'w') as f:
        f.write(str(massmax))

def set_R_RG15(z, M, mu_min):
    """
    Return RG-15 merger rate given subhalo redshift, mass, and ratio.

    Parameters
    ----
    z : float
        Redshift.
    M : float
        Subhalo mass x 10^10 Msun.
    mu_min : float
        Minimum merger subhalo mass ratio.

    Returns
    ----
    val : float
        Merger rate in 1/Gyr.
    """
    A0 = 10**-2.2287
    eta = 2.4644
    alpha0 = 0.2241
    alpha1 = -1.1759
    beta0 = -1.2595
    beta1 = 0.0611
    gamma = -0.0477
    delta0 = 0.7668
    delta1 = -0.4695
    M0 = 20

    Az = A0 * (1 + z)**eta
    alpha_z = alpha0 * (1 + z)**alpha1
    beta_z = beta0 * (1 + z)**beta1
    delta_z = delta0 * (1 + z)**delta1
    
    R = (Az * M**alpha_z * (1 + (M / M0)**delta_z)
         * (1 - mu_min**(beta_z + gamma * math.log10(M) + 1))
         / (beta_z + gamma * math.log10(M) + 1))
    return R

def setbinlims(binsmin, binsmax, numbins, logspace, mrglst3):
    """Return bin edges, centers, and widths.

    Parameters
    --------
    binsmin : int
        Minimum edge value over all bins
    binsmax : float
        Maximum edge value over all bins
    numbins : int
        Number of bins
    logspace : boolean
        Bin widths equal in logspace if true, in linspace if false
    mrglst3 : boolean
        Combine last three bins into one if true, do not if false

    Returns
    --------
    Tuple : (ndarray, ndarray, ndarray)
        Numpy arrays of the bin edges, centers, and widths
    """

    if logspace:
        edgesraw = np.logspace(math.log10(binsmin), math.log10(binsmax),
                               numbins+1, dtype=np.float_)
    else:
        edgesraw = np.linspace(binsmin, binsmax, numbins+1, dtype=np.float_)
    logging.debug(f'edgesraw {edgesraw}')
    if mrglst3:
        mask = np.ones(len(edgesraw), dtype=np.bool_)
        mask[[-3, -2]] = False
        edges = edgesraw[mask]
    else:
        edges = edgesraw
    ctrs = np.zeros(len(edges)-1)
    widths = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        widths[i] = edges[i+1] - edges[i]
        if logspace:
            ctrs[i] = 10**((math.log10(edges[i]) + math.log10(edges[i+1]))/2)
        else:
            ctrs[i] = (edges[i] + edges[i+1]) / 2
    logging.debug(f'edges {edges}\nctrs {ctrs}\nwidths {widths}')
    return (edges, ctrs, widths)

def setfs():
    """
    Get merger fraction info, store in files.
    """
    print(f'Running setfs subhalostart {cfg.subhalostart} subhalo_end '
          f'{cfg.subhalo_end}')

    # get aggregate data
    numsnapnums = 0
    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            numsnapnums += len(cfg.snapnumsOG)
        elif ilnum == 100 or ilnum == 300:
            numsnapnums += len(cfg.snapnumsTNG)

    # get mass max
    fcfg_mmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'mmax',
                           'massmax' + fcfg_mmax + '.txt')) as f:
        massmax = float(f.read())
    logging.debug(f'massmax {massmax}')

    # create arrays
    logging.debug('m bins')
    m_edges, m_ctrs, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                    cfg.mbinsnumraw, cfg.mlogspace,
                                    cfg.mmrglst3)
    numbins = len(m_ctrs)
    
    logging.debug('R-G function bins')
    RGm_edges, _, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                 cfg.RGms_num, cfg.mlogspace, 0)
    
    lenfcns = (numsnapnums * len(cfg.mu_maxes) * len(cfg.Trefs)
               * len(cfg.Tfacs))
    dtypefcns = [('ilnum', np.int16), ('snapnum', np.int16),
                 ('mu_min', np.float16), ('mu_max', np.float16),
                 ('Tref', np.unicode_, 10), ('Tfac', np.float16),
                 ('fsRGfcn', np.float_, len(RGm_edges))]
    fs_fcn_all = np.zeros(lenfcns, dtype=dtypefcns)
    lencts = (numsnapnums * len(cfg.mu_maxes) * len(cfg.virtualprogs)
              * len(cfg.SubLink_gals) * len(cfg.Trefs) * len(cfg.Tfacs))
    dtypects = [('ilnum', np.int16), ('snapnum', np.int16),
                ('mu_min', np.float16), ('mu_max', np.float16),
                ('virtualprog', np.bool_), ('SubLink_gal', np.bool_),
                ('Tref', np.unicode_, 10), ('Tfac', np.float16),
                ('numgxy', np.int_, numbins),
                ('numsame_tmrgrs', np.int_, numbins),
                ('sumprobsbin', np.float_, numbins),
                ('sumprobsmlt', np.float_, numbins),
                ('sumprobs1', np.float_, numbins), 
                ('sumprobs2', np.float_, numbins),
                ('sumprobs3', np.float_, numbins),
                ('sumprobs4', np.float_, numbins),
                ('fsRGct', np.float_, numbins),
                ('fsbin', np.float_, numbins), ('fsmlt', np.float_, numbins),
                ('fs1', np.float_, numbins), ('fs2', np.float_, numbins),
                ('fs3', np.float_, numbins), ('fs4', np.float_, numbins)]
    fs_ct_all = np.zeros(lencts, dtype=dtypects)
    fsRGfcn = np.zeros(len(RGm_edges))
    sh_cts_by_m = np.zeros(numbins)
    numsame_tmrgrs = np.zeros(numbins)
    sumprobsbin = np.zeros(numbins)
    sumprobsmlt = np.zeros(numbins)
    sumprobs1 = np.zeros(numbins)
    sumprobs2 = np.zeros(numbins)
    sumprobs3 = np.zeros(numbins)
    sumprobs4 = np.zeros(numbins)
    fsbin = np.zeros(numbins)
    fsmlt = np.zeros(numbins)
    fs1 = np.zeros(numbins)
    fs2 = np.zeros(numbins)
    fs3 = np.zeros(numbins)
    fs4 = np.zeros(numbins)
    fsRGct = np.zeros(numbins)
    i_fcn = 0
    i_ct = 0
    T = 0
    logging.debug(f'numsnapnums {numsnapnums} lenfcns {lenfcns} '
                  f'lencts {lencts}')

    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        dictnum, _, ilrun, _, _, _ = get_run_info(ilnum)

        for j, k, p, u in it.product(snapnums, cfg.mu_maxes, cfg.Trefs,
                                     cfg.Tfacs):
            logging.debug(f'ilnum {ilnum} snapnum {j} mu_max {k} Tref {p} '
                          f'Tfac {u}')

            # get ta related data
            z = glb.zs[dictnum][j]
            ta = glb.ts[dictnum][j]
            tam1 = glb.ts[dictnum][j-1]
            mu_min = 1/k
            if p == 'analysis' or p == 'merger':
                T = glb.Tsnys[dictnum][j] * u
            elif p == 'snapwidth':
                T = glb.Tsnps[dictnum][j] * u
            logging.debug(f'z {z} ta {ta} tam1 {tam1} T {T}')

            # get RG function-derived fractions
            for q in range(len(RGm_edges)):
                rateRG = set_R_RG15(z, RGm_edges[q], mu_min)
                fsRGfcn[q] = rateRG * T
                logging.debug(f'M {RGm_edges[q]} rateRG {rateRG} '
                              f'fsRGfcn[q] {fsRGfcn[q]}')
            logging.debug(f'fsRGfcn {fsRGfcn}')
            fs_fcn_all[i_fcn] = ((ilnum, j, mu_min, k, p, u, fsRGfcn))
            i_fcn += 1

            for n, o in it.product(cfg.virtualprogs, cfg.SubLink_gals):
                print(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                      f'virtualprogs {n} SubLink_gals {o} Tref {p} Tfac {u}')
                logging.debug(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                              f'virtualprogs {n} SubLink_gals {o} Tref {p} '
                              f'Tfac {u} T {T} z {z} ta {ta} tam1 {tam1}')

                sh_cts_by_m.fill(0)
                sumprobsbin.fill(0); sumprobsmlt.fill(0)
                sumprobs1.fill(0); sumprobs2.fill(0)
                sumprobs3.fill(0); sumprobs4.fill(0)
                fsRGct.fill(-1); fsbin.fill(-1); fsmlt.fill(-1)
                fs1.fill(-1); fs2.fill(-1)
                fs3.fill(-1); fs4.fill(-1)

                # get merger data
                f_cfg_mgr = (f'i{ilnum}s{j}rl{mu_min:1.2f}ru{k:02d}v{n}'
                             f'g{o}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                             f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
                with np.load(os.path.join('output', 'numerical', 'mrgr',
                                          ilrun, 'mrgrdat' + f_cfg_mgr
                                          + '.npz')) as data:
                    subhalos = data['subhalos']
                    mergers = data['mergers']
                    mrgrs_same_t = data['mrgrs_same_t']
                logging.debug(f'subhalos\n{subhalos}\nmrgrs_same_t\n'
                              f'{mrgrs_same_t}\nmergers\n{mergers}')

                # bin subhalos and same_t mergers
                m_inds_by_sh = np.digitize(subhalos['m'], m_edges) - 1
                numsame_tmrgrs = np.histogram(mrgrs_same_t, m_edges)[0]
                logging.debug(f'm_inds_by_sh\n{m_inds_by_sh}\n'
                              f'numsame_tmrgrs\n{numsame_tmrgrs}')
                
                # get per-subhalo fractions
                for q in range(len(subhalos)):
                    logging.debug(f'subhalos[q] {subhalos[q]}')
                    mrgr_snps_sh = mergers['snap_m'][
                            (mergers['s_id_a'] == subhalos['s_id'][q])]
                    logging.debug(f'mrgr_snps_sh {mrgr_snps_sh}')
                    
                    # no mergers associated with this subhalo
                    if len(mrgr_snps_sh) == 0:
                        continue
            
                    pbin, pmlt = 0, 0
                    exact_p_tot = np.zeros(4)
                    probs = []
                    
                    # set probabilities
                    for mrgr_snp in mrgr_snps_sh:
                        tms = glb.ts[dictnum][mrgr_snp-1]
                        tme = glb.ts[dictnum][mrgr_snp]
                        if p == 'merger':
                            T = glb.Tsnys[dictnum][mrgr_snp] * u
                        logging.debug(f'tms {tms} tme {tme} T {T}')
            
                        if tme <= ta:
                            prob = (T/2 - (ta - tme)) / (tme - tms)
                        else:
                            prob = (T/2 - (tms - ta)) / (tme - tms)
                        logging.debug(f'prob {prob}')
                        if prob > 1:
                            probs.append(1)
                        elif prob < 0:
                            probs.append(0)
                        else:
                            probs.append(prob)
                    logging.debug(f'probs {probs}')
            
                    # > 4 p = 1's: p tables not needed
                    if probs.count(1) > 4:
                        pbin, pmlt = 1, 1
                        exact_p_tot.fill(0)
                        logging.debug('count (probs == 1) > 4')
                        
                    # create p tables
                    else:
                        # remove p = 0 subhalos
                        probs = [prb for prb in probs if prb != 0]
                        numprobs = len(probs)
                        logging.debug(f'post-zero-removal probs {probs}')
            
                        if numprobs == 0:
                            logging.debug('no nonzero probs; skip table')
                            continue
                        
                        # get probs for exactly a certain # of mergers
                        for r in range(1, 5):
                            logging.debug(f'exactly {r} mergers')
                            if r == 1:
                                bools = np.identity(numprobs)
                                numtrues = numprobs
                                bxp = np.zeros(numprobs)
                            elif numprobs >= r:
                                trues = list(it.combinations(range(numprobs),
                                                             r))
                                numtrues = len(trues)
                                bools = np.zeros((numtrues, numprobs))
                                logging.debug(f'trues {trues}')
                                for s in range(numtrues):
                                    for v in range(numprobs):
                                        for w in range(r):
                                            if trues[s][w] == v:
                                                bools[s][v] = 1
                            else:
                                bools = np.array([])
                            logging.debug(f'bools {bools}')
                                
                            # get joint probabilities
                            if numprobs >= r:
                                for s in range(numtrues):
                                    for v in range(numprobs):
                                        if bools[s][v] == 1:
                                            bxp[v] = probs[v]
                                        else:
                                            bxp[v] = 1 - probs[v]
                                    incval = np.prod(bxp)
                                    exact_p_tot[r-1] += incval
                                    logging.debug(
                                            f'{r} bxp {bxp} incval {incval} '
                                            f'exact_p_tot {exact_p_tot}')
                        
                        # at least 1 merger
                        logging.debug('binary mergers')
                        for r in range(numprobs):
                            bxp[r] = 1 - probs[r]
                        pbin = 1 - np.prod(bxp)
                        logging.debug(f'bin bxp {bxp} pbin {pbin}')
                        
                        # at least 2 mergers
                        logging.debug('multiple mergers')
                        if numprobs >= 2:
                            pmlt = pbin - exact_p_tot[0]
                        logging.debug(f'mlt pbin {pbin} exact_p_tot[0] '
                                      f'{exact_p_tot[0]} pmlt {pmlt}')
                            
                    sumprobsbin[m_inds_by_sh[q]] += pbin
                    sumprobsmlt[m_inds_by_sh[q]] += pmlt
                    sumprobs1[m_inds_by_sh[q]] += exact_p_tot[0]
                    sumprobs2[m_inds_by_sh[q]] += exact_p_tot[1]
                    sumprobs3[m_inds_by_sh[q]] += exact_p_tot[2]
                    sumprobs4[m_inds_by_sh[q]] += exact_p_tot[3]
                    logging.debug(
                            f'sumprobsbin {sumprobsbin}\n'
                            f'sumprobsmlt {sumprobsmlt}\n'
                            f'sumprobs1 {sumprobs1}\nsumprobs2 {sumprobs2}\n'
                            f'sumprobs3 {sumprobs3}\nsumprobs4 {sumprobs4}')

                # calculate/store data
                sh_cts_by_m = np.bincount(m_inds_by_sh, minlength=len(m_ctrs))
                if p == 'merger':
                    T = glb.Tsnys[dictnum][j] * u
                logging.debug(f'T_fsRGct p {p} u {u} T {T}')
                for q in range(len(m_ctrs)):
                    if sh_cts_by_m[q] > 0:
                        fsbin[q] = sumprobsbin[q] / sh_cts_by_m[q]
                        fsmlt[q] = sumprobsmlt[q] / sh_cts_by_m[q]
                        fs1[q] = sumprobs1[q] / sh_cts_by_m[q]
                        fs2[q] = sumprobs2[q] / sh_cts_by_m[q]
                        fs3[q] = sumprobs3[q] / sh_cts_by_m[q]
                        fs4[q] = sumprobs4[q] / sh_cts_by_m[q]
                        fsRGct[q] = (numsame_tmrgrs[q] / sh_cts_by_m[q] /
                                     (ta - tam1) * T)
                logging.debug(f'sh_cts_by_m {sh_cts_by_m}\nfsbin {fsbin}\n'
                              f'fsmlt {fsmlt}\nfs1 {fs1}\nfs2 {fs2}\n'
                              f'fs3 {fs3}\nfs4 {fs4}\nfsRGct {fsRGct}')
                
                fs_ct_all[i_ct] = ((
                        ilnum, j, mu_min, k, n, o, p, u, sh_cts_by_m,
                        numsame_tmrgrs, sumprobsbin, sumprobsmlt, sumprobs1,
                        sumprobs2, sumprobs3, sumprobs4, fsRGct, fsbin, fsmlt,
                        fs1, fs2, fs3, fs4))
                logging.debug(f'fs_ct_all\n{fs_ct_all}\ni_ct {i_ct}')
                i_ct += 1
    logging.debug(f'final: fs_fcn_all\n{fs_fcn_all}\nfs_ct_all\n{fs_ct_all}')

    # save results
    pathname = os.path.join('output', 'numerical', 'f')
    if (not os.path.exists(pathname)):
        os.makedirs(pathname)
    fcfg_f = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
              f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
              f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
              f'Rn{cfg.RGms_num}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    np.savez_compressed(os.path.join(pathname, 'fdat' + fcfg_f + '.npz'),
                        fs_fcn_all=fs_fcn_all, fs_ct_all=fs_ct_all)

def viewfvmdata():
    """
    View f_vs_m raw output data.
    """

    print('Running viewfvmdata')

    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    fcfg_f = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
              f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
              f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
              f'Rn{cfg.RGms_num}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with np.load(os.path.join('output', 'numerical', 'f',
                              'fdat' + fcfg_f + '.npz')) as data:
        with open('fdat' + fcfg_f + '.txt', 'w') as f:
            for i in data:
                f.write(i + '\n')
                for j in range(len(data[i])):
                    f.write(str(data[i][j]) + '\n')

def setdtmax():
    """
    Compute maximum dt for a set of mergers.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    print(f'Running setdtmax subhalostart {cfg.subhalostart} '
          f'subhalo_end {cfg.subhalo_end}')

    dtmax = 0

    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        dictnum, basepath, ilrun, snapnummax, _, _ = get_run_info(ilnum)

        for j, k, n, o, p, u in it.product(snapnums, cfg.mu_maxes,
                                           cfg.virtualprogs, cfg.SubLink_gals,
                                           cfg.Trefs, cfg.Tfacs):
            print(f'Analyzing ilnum {ilnum} snapnum {j} mu_max {k} '
                  f'virtualprog {n} SubLink_gal {o} Tref {p} Tfac {u}')
            logging.debug(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                          f'virtualprog {n} SubLink_gal {o} Tref {p} Tfac {u}')

            # load merger data
            mu_min = 1/k
            f_cfg_mgr = (f'i{ilnum}s{j}rl{mu_min:1.2f}ru{k:02d}v{n}'
                         f'g{o}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                         f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'mrgr', ilrun,
                                      'mrgrdat' + f_cfg_mgr + '.npz')) as data:
                subhalos = data['subhalos']
                mergers = data['mergers']
            logging.debug(f'subhalos\n{subhalos}\nmergers\n{mergers}')
            
            for q in range(len(subhalos)):
                mrgrsnps = mergers[(mergers['s_id_a'] == 
                                    subhalos[q]['s_id'])]['snap_m']
                logging.debug(f'subhalo {q} mrgrsnps {mrgrsnps}')
                for mrgrsnp in mrgrsnps:
                    # since merger happened between tms and tme
                    if mrgrsnp <= j:
                        dt = glb.ts[dictnum][j] - glb.ts[dictnum][mrgrsnp-1]
                    else:
                        dt = glb.ts[dictnum][mrgrsnp] - glb.ts[dictnum][j]
                    if dt > dtmax:
                        dtmax = dt
                    logging.debug(f'mrgrsnp {mrgrsnp} anlys_snp {j} dt {dt} '
                                  f'dtmax {dtmax}')

    # display/store dt max
    logging.debug(f'dt max {dtmax}')
    print(f'dt max {dtmax}')
    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    pathname = os.path.join('output', 'numerical', 'dtmax')
    if (not os.path.exists(pathname)):
        os.makedirs(pathname)
    fcfg_dtmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                  f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                  f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join(pathname, 'dtmax' + fcfg_dtmax + '.txt'), 'w') as f:
        f.write(str(dtmax))

def set_dt_incs(dictnum, snapa, mrgrs, dt_edges, clst):
    """
    Compute dt increment values.

    Parameters
    ----
    dictnum : int
        config.py dictionary number: 0: Illustris; 1: TNG
    snapa : int
        Snapshot number of analysis.
    mrgrs : numpy.ndarray(('snapms', np.int16), ('tms', np.float_),
                          ('snapme', np.int16), ('tme', np.float_), 
                          ('prob', np.float_))
        snapnum and time of each merger's start and end, and overlap
        probability.
    dt_edges : numpy.ndarray((np.float_))
        Edges of the dt bins to increment.
    clst : boolean
        If True, closest mergers analyzed; if False, previous/next
        mergers.

    Returns
    ----
    tuple (numpy.array, numpy.array)
        If clst, bin increments for dts_c0n0 and dts_c1yc2n, if not, for
        dts1p and dts2p, or dts1n and dts2n.
    """
    ta = glb.ts[dictnum][snapa]
    dtbinsnum = len(dt_edges) - 1
    num_mrgrs = len(mrgrs)
    num_mrgrs_p0 = len(mrgrs[mrgrs['prob'] == 0])
    num_mrgrs_ppos = num_mrgrs - num_mrgrs_p0
    logging.debug(f'ta {ta} num_mrgrs {num_mrgrs} num_mrgrs_p0 {num_mrgrs_p0} '
                  f'num_mrgrs_ppos {num_mrgrs_ppos}')

    if num_mrgrs_p0 > 0:
        dtsp0 = np.full(num_mrgrs_p0, -1, dtype=np.float_)
    if num_mrgrs_ppos > 0:
        mrgrs_ppos = np.full(num_mrgrs_ppos, -1, dtype=mrgrs.dtype)
    dt1, dt2 = -1, -1
    i_p0, i_ppos = 0, 0

    if clst:
        dts_c0n0_inc = np.zeros((dtbinsnum, dtbinsnum))
        dts_c1yc2n_inc = np.zeros((dtbinsnum))
    else:
        dts1inc = np.zeros((dtbinsnum))
        dts2inc = np.zeros((dtbinsnum))

    # fill p0, ppos bins
    for i in range(num_mrgrs):
        if mrgrs['prob'][i] == 0:
            dtsp0[i_p0] = abs(mrgrs['tme'][i] - ta)
            logging.debug(f"p = 0: tme {mrgrs['tme'][i]} ta {ta}")
            i_p0 += 1
        else:
            mrgrs_ppos[i_ppos] = mrgrs[i]
            i_ppos += 1
    logging.debug(f'mrgrs {mrgrs}')
    if num_mrgrs_p0 > 0:
        logging.debug(f'dtsp0 {dtsp0}')
    if num_mrgrs_ppos > 0:
        logging.debug(f'mrgrs_ppos {mrgrs_ppos}')

    # no mergers with p > 0
    if num_mrgrs_ppos == 0:
        logging.debug('num_mrgrs_ppos == 0:')

        # determine dt1, dt2
        dt1 = np.partition(dtsp0, 0)[0]
        if num_mrgrs_p0 >= 2:
            dt2 = np.partition(dtsp0, 1)[1]

		# increment arrays
        if clst:
            if dt2 == -1:
                dts_c1yc2n_inc[np.digitize(dt1, dt_edges)-1] += 1
            else:
                dts_c0n0_inc[np.digitize(dt2, dt_edges)-1]\
                        [np.digitize(dt1, dt_edges)-1] += 1
        else:
            dts1inc[np.digitize(dt1, dt_edges)-1] += 1
            if dt2 != -1:
                try:
                    dts2inc[np.digitize(dt2, dt_edges)-1] += 1
                except IndexError:
                    print(dt2, dt_edges)

    # at least one merger with p > 0
    else:
        logging.debug('num_mrgrs_ppos > 0')

        # check whether p table can be avoided, i.e. all p > 0 mergers are
        # p == 1 and abs(tme - ta) == 0
        p_tbl_rqd = False
        for i in range(num_mrgrs_ppos):
            p_lt_1 = mrgrs_ppos['prob'][i] < 1
            tme_eq_ta = mrgrs_ppos['snapme'][i] == snapa
            p_tbl_rqd = p_lt_1 or not tme_eq_ta
            logging.debug(f'i {i} p < 1 {p_lt_1} tme = ta {tme_eq_ta} '
                          f'p_tbl_rqd {p_tbl_rqd}')
            if p_tbl_rqd: 
                break

        # probability table not required
        if not p_tbl_rqd:
            logging.debug('all p > 0 mergers p == 1, dt == 0')
            if clst:
                if num_mrgrs_ppos == 1:
                    if num_mrgrs_p0 == 0:
                        dts_c1yc2n_inc[0] += 1
                    else:
                        dt2 = np.partition(dtsp0, 0)[0]
                        dts_c0n0_inc[np.digitize(dt2, dt_edges)-1][0] += 1
                else:
                    dts_c0n0_inc[0][0] += 1
            else:
                dts1inc[0] += 1
                if num_mrgrs_ppos == 1 and num_mrgrs_p0 > 0:
                    dt2 = np.partition(dtsp0, 0)[0]
                    dts2inc[np.digitize(dt2, dt_edges)-1] += 1
                elif num_mrgrs_ppos >= 2:
                    dts2inc[0] += 1

        # p table required   
        else:
            # test if table can be truncated
            trunc_p_tbl = False
            if num_mrgrs_ppos >= 4:
                mrgrcts = np.bincount(mrgrs_ppos['snapme'])
                logging.debug(f'num_mrgrs_ppos >= 4: mrgrcts {mrgrcts}')
                if np.amax(mrgrcts) >= 4:
                    trunc_p_tbl = True
            
            # large number of identical-z mergers; truncate p table
            if trunc_p_tbl == True:
                snaprmvd = np.argmax(mrgrcts)
                num_mrgrs_rmvd = mrgrcts[snaprmvd]
                mrgrs_p_tbl = mrgrs_ppos[mrgrs_ppos['snapme'] != snaprmvd]
                mrgrs_rmvd_dat = (mrgrs_ppos[mrgrs_ppos['snapme']
                                  == snaprmvd][0])
                if snaprmvd <= snapa:
                    dtmin_mrgrs_rmvd = ta - mrgrs_rmvd_dat['tme']
                    dtmax_mrgrs_rmvd = ta - mrgrs_rmvd_dat['tms']
                else:
                    dtmin_mrgrs_rmvd = mrgrs_rmvd_dat['tms'] - ta
                    dtmax_mrgrs_rmvd = mrgrs_rmvd_dat['tme'] - ta
            else:
                snaprmvd = None
                num_mrgrs_rmvd = 0
                mrgrs_p_tbl = mrgrs_ppos
                mrgrs_rmvd_dat = None
                dtmin_mrgrs_rmvd = None
                dtmax_mrgrs_rmvd = None
            logging.debug(f'snaprmvd {snaprmvd} '
                          f'num_mrgrs_rmvd {num_mrgrs_rmvd}\n'
                          f'mrgrs_p_tbl {mrgrs_p_tbl}\n'
                          f'mrgrs_rmvd_dat {mrgrs_rmvd_dat}\n'
                          f'dtmin_mrgrs_rmvd {dtmin_mrgrs_rmvd} '
                          f'dtmax_mrgrs_rmvd {dtmax_mrgrs_rmvd}')

            # fill p table                
            num_mrgrs_p_tbl = len(mrgrs_p_tbl)
            bools = np.zeros(num_mrgrs_p_tbl)
            bxp = np.zeros(num_mrgrs_p_tbl)
            for i in range(2**num_mrgrs_p_tbl):

                # create probability table row
                bin_num = (format(i, 'b').zfill(num_mrgrs_p_tbl))
                dts_p_tbl = [0] * num_mrgrs_p_tbl
                
                # set p table row values
                for j in range(num_mrgrs_p_tbl):
                    bools[j] = bin_num[j]
                    if bools[j] == 1:
                        bxp[j] = mrgrs_p_tbl['prob'][j]
                        if mrgrs_p_tbl['tme'][j] <= ta:
                            dts_p_tbl[j] = ta - mrgrs_p_tbl['tme'][j]
                        else:
                            dts_p_tbl[j] = mrgrs_p_tbl['tms'][j] - ta
                    else:
                        bxp[j] = 1 - mrgrs_p_tbl['prob'][j]
                        if mrgrs_p_tbl['tme'][j] <= ta:
                            dts_p_tbl[j] = ta - mrgrs_p_tbl['tms'][j]
                        else:
                            dts_p_tbl[j] = mrgrs_p_tbl['tme'][j] - ta
                logging.debug(f'bin_num {bin_num} i {i} bools {bools}\n'
                              f'bxp {bxp}\ndts_p_tbl {dts_p_tbl}')

                if trunc_p_tbl == True:
                    # set zero-true values
                    dts_p_tbl0T = np.append(dts_p_tbl, [dtmax_mrgrs_rmvd,
                                            dtmax_mrgrs_rmvd])
                    dts_p_tbl0Tmins = np.partition(dts_p_tbl0T, (0,1))
                    dt_p_tbl0T1 = dts_p_tbl0Tmins[0]
                    dt_p_tbl0T2 = dts_p_tbl0Tmins[1]
                    incval0T = (np.prod(bxp)
                                * (1 - mrgrs_rmvd_dat['prob'])**num_mrgrs_rmvd)
                    logging.debug(f'dts_p_tbl0T {dts_p_tbl0T}\n'
                                  f'dts_p_tbl0Tmins {dts_p_tbl0Tmins}\n'
                                  f'dt_p_tbl0T1 {dt_p_tbl0T1} '
                                  f'dt_p_tbl0T2 {dt_p_tbl0T2} '
                                  f'incval0T {incval0T}')
                    
                    # add zero-true increments
                    if clst:
                        dts_c0n0_inc[np.digitize(dt_p_tbl0T2, dt_edges)-1] \
                                [np.digitize(dt_p_tbl0T1, dt_edges)-1] \
                                += incval0T
                        logging.debug(f'dts_c0n0_inc {dts_c0n0_inc}')
                    else:
                        dts1inc[np.digitize(dt_p_tbl0T1, dt_edges)-1] \
                                += incval0T
                        dts2inc[np.digitize(dt_p_tbl0T2, dt_edges)-1] \
                                += incval0T
                        logging.debug(f'dts1inc {dts1inc}\ndts2inc {dts2inc}')
                    
                    
                    # set one-true values
                    dts_p_tbl1T = np.append(dts_p_tbl, [dtmin_mrgrs_rmvd, 
                                            dtmax_mrgrs_rmvd])
                    dts_p_tbl1Tmins = np.partition(dts_p_tbl1T, (0,1))
                    dt_p_tbl1T1 = dts_p_tbl1Tmins[0]
                    dt_p_tbl1T2 = dts_p_tbl1Tmins[1]
                    incval1T = (np.prod(bxp) * mrgrs_rmvd_dat['prob'] 
                                * (1 - mrgrs_rmvd_dat['prob'])
                                    **(num_mrgrs_rmvd - 1)
                                * num_mrgrs_rmvd)
                    logging.debug(f'dts_p_tbl1T {dts_p_tbl1T}\n'
                                  f'dts_p_tbl1Tmins {dts_p_tbl1Tmins}\n'
                                  f'dt_p_tbl1T1 {dt_p_tbl1T1} '
                                  f'dt_p_tbl1T2 {dt_p_tbl1T2} '
                                  f'incval1T {incval1T}')

                    # add one-true increments                    
                    if clst:
                        dts_c0n0_inc[np.digitize(dt_p_tbl1T2, dt_edges)-1] \
                                [np.digitize(dt_p_tbl1T1, dt_edges)-1] \
                                += incval1T
                        logging.debug(f'dts_c0n0_inc {dts_c0n0_inc}')
                    else:
                        dts1inc[np.digitize(dt_p_tbl1T1, dt_edges)-1] \
                                += incval1T
                        dts2inc[np.digitize(dt_p_tbl1T2, dt_edges)-1] \
                                += incval1T
                        logging.debug(f'dts1inc {dts1inc}\ndts2inc {dts2inc}')

                    
                    # set >= 2-true dts
                    dts_p_tbl_ge2T = np.append(dts_p_tbl, [dtmin_mrgrs_rmvd, 
                                               dtmin_mrgrs_rmvd])
                    dts_p_tbl_ge2Tmins = np.partition(dts_p_tbl_ge2T, (0,1))
                    dt_p_tbl_ge2T1 = dts_p_tbl_ge2Tmins[0]
                    dt_p_tbl_ge2T2 = dts_p_tbl_ge2Tmins[1]
                    logging.debug(f'dts_p_tbl_ge2T {dts_p_tbl_ge2T}\n'
                                  f'dts_p_tbl_ge2Tmins {dts_p_tbl_ge2Tmins}\n'
                                  f'dt_p_tbl_ge2T1 {dt_p_tbl_ge2T1} '
                                  f'dt_p_tbl_ge2T2 {dt_p_tbl_ge2T2}')

                    # set, add >= 2-true increments
                    for j in range(2, num_mrgrs_rmvd+1):
                        incval_ge2T = (np.prod(bxp)
                                       * mrgrs_rmvd_dat['prob']**(j)
                                       * (1 - mrgrs_rmvd_dat['prob'])
                                           **(num_mrgrs_rmvd - j)
                                       * math.factorial(num_mrgrs_rmvd)
                                       / math.factorial(j)
                                       / math.factorial(num_mrgrs_rmvd - j))
                        logging.debug(f'j {j} incval_ge2T {incval_ge2T}')
                        if clst:
                            dts_c0n0_inc[np.digitize(dt_p_tbl_ge2T2,
                                                     dt_edges)-1] \
                                    [np.digitize(dt_p_tbl_ge2T1, dt_edges)-1] \
                                    += incval_ge2T
                            logging.debug(f'j {j} dts_c0n0_inc {dts_c0n0_inc}')
                        else:
                            dts1inc[np.digitize(dt_p_tbl_ge2T1, dt_edges)-1] \
                                    += incval_ge2T
                            dts2inc[np.digitize(dt_p_tbl_ge2T2, dt_edges)-1] \
                                    += incval_ge2T
                            logging.debug(f'j {j} dts1inc {dts1inc}\n'
                                          f'dts2inc {dts2inc}')    
    
                else:
                    incval = np.prod(bxp)
                    dt1 = np.partition(dts_p_tbl, 0)[0]
                    if num_mrgrs_ppos >= 2:
                        dt2 = np.partition(dts_p_tbl, 1)[1]
                    elif num_mrgrs_p0 >= 1:
                        dt2 = np.partition(dtsp0, 0)[0]
                    logging.debug(f'incval {incval} dt1 {dt1} dt2 {dt2}')

                    if clst:
                        if dt2 == -1:
                            dts_c1yc2n_inc[np.digitize(dt1, dt_edges)-1]\
                                += incval
                        else:
                            dts_c0n0_inc[np.digitize(dt2, dt_edges)-1]\
                                [np.digitize(dt1, dt_edges)-1] += incval
                    else:
                        dts1inc[np.digitize(dt1, dt_edges)-1] += incval
                        if dt2 != -1:
                            dts2inc[np.digitize(dt2, dt_edges)-1] += incval
    if clst:
        logging.debug(f'dts_c0n0_inc\n{dts_c0n0_inc}\n'
                      f'dts_c1yc2n_inc {dts_c1yc2n_inc}')
        return(dts_c0n0_inc, dts_c1yc2n_inc)
    else:
        logging.debug(f'dts1inc {dts1inc}\ndts2inc {dts2inc}')
        return(dts1inc, dts2inc)


def setdts():
    """
    Calculate dt info across all data generated by a given set of
    configuration parameters.
    """
    print(f'Running setdts subhalostart {cfg.subhalostart} subhalo_end '
          f'{cfg.subhalo_end}')

    # confirm correct input parameters
    if not isinstance(cfg.KDEmult, int) or cfg.KDEmult < 1:
        raise ValueError('KDEmult must be an integer at least 1')

    # set aggregate data
    numsnapnums = 0
    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            numsnapnums += len(cfg.snapnumsOG)
        elif ilnum == 100 or ilnum == 300:
            numsnapnums += len(cfg.snapnumsTNG)
    logging.debug(f'num snapnums {numsnapnums}')

    # get mass max and set bin info
    logging.debug('m bins')
    fcfg_mmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'mmax',
                           'massmax' + fcfg_mmax + '.txt')) as f:
              massmax = float(f.read())
    logging.debug(f'massmax {massmax}')
    m_edges, m_ctrs, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                    cfg.mbinsnumraw, cfg.mlogspace,
                                    cfg.mmrglst3)
    mbinsnum = len(m_ctrs)

    # get dtmax
    logging.debug('dt bins')
    fcfg_dtmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                  f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                  f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'dtmax',
                           'dtmax' + fcfg_dtmax + '.txt')) as f:
              dtmax = float(f.read())
    logging.debug(f'dtmax {dtmax}')
    
    dtype_mrgr_info = [('snapms', np.int16), ('tms', np.float_),
                       ('snapme', np.int16), ('tme', np.float_), 
                       ('prob', np.float_)]

    # get f data
    fcfg_f = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
              f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
              f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
              f'Rn{cfg.RGms_num}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with np.load(os.path.join('output', 'numerical', 'f',
                              'fdat' + fcfg_f + '.npz')) as data:
        fs_ct_all = data['fs_ct_all']
    logging.debug(f'fs_ct_all\n{fs_ct_all.dtype.names}\n{fs_ct_all}')

    # set dts
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        dictnum, basepath, ilrun, snapnummax, _, _ = get_run_info(ilnum)

        pathname = os.path.join('output', 'numerical', 'dt', ilrun)
        if (not os.path.exists(pathname)):
            os.makedirs(pathname)

        for j, k, n, o, p, u in it.product(snapnums, cfg.mu_maxes,
                                           cfg.virtualprogs, cfg.SubLink_gals,
                                           cfg.Trefs, cfg.Tfacs):
            print(f'Analyzing ilnum {ilnum} snapnum {j} mu_max {k} '
                  f'virtualprog {n} SubLink_gal {o} Tref {p} Tfac {u}')
            logging.debug(f'ilnum {ilnum} snapnum {j} mu_max {k} '
                          f'virtualprog {n} SubLink_gal {o} Tref {p} Tfac {u}')
            
            # move to next parameter configuration if file already generated            
            filename = (f'dt_dati{ilnum}s{j}r{k}v{n}g{o}Tr{p[0]}Tf{u}'
                        f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                        f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
                        f'Rn{cfg.RGms_num}do{cfg.dtbinwdthopt:1.1f}'
                        f'fn{cfg.fxs_num}Km{cfg.KDEmult}ss{cfg.subhalostart}'
                        f'se{cfg.subhalo_end}.npz')
            if os.path.isfile(os.path.join(pathname, filename)):
                print('File already created, moving to next parameter '
                      'configuration')
                logging.debug('File already created, moving to next parameter '
                              'configuration')
                continue

            # set t, T
            ta = glb.ts[dictnum][j]
            mu_min = 1/k
            if p == 'analysis' or p == 'merger':
                T = glb.Tsnys[dictnum][j] * u
            elif p == 'snapwidth':
                T = glb.Tsnps[dictnum][j] * u
            logging.debug(f'ta {ta} T {T}')

            # set dt bin info
            dtbinwidth = T/2 / max(1, round(T/2 / cfg.dtbinwdthopt))
            dtbinsnum = math.ceil((dtmax + cfg.bin_pdng) / dtbinwidth)
            dt_edgemax = dtbinwidth * dtbinsnum
            logging.debug(f'dtbinwidth {dtbinwidth} dtbinsnum {dtbinsnum} '
                          f'dt_edgemax {dt_edgemax}')
            dt_edges, dt_ctrs, dt_wdths = setbinlims(0, dt_edgemax,
                                                     dtbinsnum, 0, 0)
            
            logging.debug('f_xs')
            _, f_xctrs, f_xwdths = setbinlims(0, dt_edgemax, cfg.fxs_num, 0, 0)

            # create arrays
            Rs_dfv0 = np.full(mbinsnum, -1, np.float_)
            Rs_dfv1 = np.full(mbinsnum, -1, dtype=np.float_)
            fs_vldc1 = np.full(mbinsnum, -1, dtype=np.float_)
            fs_vldc2 = np.full(mbinsnum, -1, dtype=np.float_)
            fsKDEc1 = np.full(mbinsnum, -1, dtype=np.float_)
            fsKDEc2 = np.full(mbinsnum, -1, dtype=np.float_)
            fsPDdfv0c1 = np.full(mbinsnum, -1, dtype=np.float_)
            fsPDdfv0c2 = np.full(mbinsnum, -1, dtype=np.float_)
            fsPDdfv1c1 = np.full(mbinsnum, -1, dtype=np.float_)
            fsPDdfv1c2 = np.full(mbinsnum, -1, dtype=np.float_)
            dts_c0n0 = np.zeros((mbinsnum, dtbinsnum, dtbinsnum))
            dts_c1yc2n = np.zeros((mbinsnum, dtbinsnum))
            dts_c1nc2n = np.zeros(mbinsnum)
            dts_c1_c0n1 = np.zeros((mbinsnum, dtbinsnum))
            dts_c1_c1n0 = np.zeros((mbinsnum, dtbinsnum))
            dts_c1_c1n1 = np.zeros((mbinsnum, dtbinsnum))
            dts_c2_c0n1 = np.zeros((mbinsnum, dtbinsnum))
            dts_c2_c1n0 = np.zeros((mbinsnum, dtbinsnum))
            dts_c2_c1n1 = np.zeros((mbinsnum, dtbinsnum))
            dts1p = np.zeros((mbinsnum, dtbinsnum))
            dts2p = np.zeros((mbinsnum, dtbinsnum))
            dts1n = np.zeros((mbinsnum, dtbinsnum))
            dts2n = np.zeros((mbinsnum, dtbinsnum))
            pdfs_dtc_dfv0c1 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_dtc_dfv0c2 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_dtc_dfv1c1 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_dtc_dfv1c2 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_Rc_dfv0c1 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_Rc_dfv0c2 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_Rc_dfv1c1 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfs_Rc_dfv1c2 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfsKDEc1 = np.zeros((mbinsnum, cfg.fxs_num))
            pdfsKDEc2 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_dtc_dfv0c1 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_dtc_dfv0c2 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_dtc_dfv1c1 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_dtc_dfv1c2 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_Rc_dfv0c1 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_Rc_dfv0c2 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_Rc_dfv1c1 = np.zeros((mbinsnum, cfg.fxs_num))
            cdfs_Rc_dfv1c2 = np.zeros((mbinsnum, cfg.fxs_num))
            dts_c0n0_inc = np.zeros(dtbinsnum)
            dts_c1yc2n_inc = np.zeros(dtbinsnum)
            dts_p1inc, dts_p2inc = np.zeros(dtbinsnum), np.zeros(dtbinsnum)
            dts_n1inc, dts_n2inc = np.zeros(dtbinsnum), np.zeros(dtbinsnum)
            dts_c1_sums = np.zeros(dtbinsnum)
            dts_c2_sums = np.zeros(dtbinsnum)

            # load merger data
            f_cfg_mgr = (f'i{ilnum}s{j}rl{mu_min:1.2f}ru{k:02d}v{n}'
                         f'g{o}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                         f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'mrgr', ilrun,
                                      'mrgrdat' + f_cfg_mgr + '.npz')) as data:
                subhalos = data['subhalos']
                mergers = data['mergers']
            logging.debug(f'subhalos\n{subhalos}\nmergers\n{mergers}')

            # place subhalos in mass bins
            shs_by_m = [[] for x in range(len(m_ctrs))]
            m_inds_by_sh = np.digitize(subhalos['m'], m_edges) - 1
            for q in range(len(subhalos)):
                shs_by_m[m_inds_by_sh[q]].append(subhalos[q])
            logging.debug(f'm_inds_by_sh\n{m_inds_by_sh}\n'
                          f'shs_by_m\n{shs_by_m}')

            for q in range(len(m_ctrs)):
                logging.debug(f'mbin_num {q}')

                if len(shs_by_m[q]) == 0:
                    logging.debug('No galaxies in this mass bin')
                    continue
    
                # compute non-f_valid-dependent probability dist values
                fRGct = fs_ct_all[(fs_ct_all['ilnum'] == ilnum)
                                  & (fs_ct_all['snapnum'] == j)
                                  & (fs_ct_all['mu_min'] == mu_min)
                                  & (fs_ct_all['mu_max'] == k)
                                  & (fs_ct_all['virtualprog'] == n)
                                  & (fs_ct_all['SubLink_gal'] == o)
                                  & (fs_ct_all['Tref'] == p)
                                  & (fs_ct_all['Tfac'] == u)]['fsRGct'][0][q]
                Rs_dfv0[q] = fRGct / T
                # multiply R by 2 in below since Poisson distributions are
                # based on time to *next* event, and we need time to *closest* 
                if Rs_dfv0[q] > 0:
                    pdfs_Rc_dfv0c1[q] = stats.expon.pdf(
                            f_xctrs, scale=1/(2*Rs_dfv0[q]))
                    pdfs_Rc_dfv0c2[q] = stats.erlang.pdf(
                            f_xctrs, 2, scale=1/(2*Rs_dfv0[q]))
                    cdfs_Rc_dfv0c1[q] = stats.expon.cdf(
                            f_xctrs, scale=1/(2*Rs_dfv0[q]))
                    cdfs_Rc_dfv0c2[q] = stats.erlang.cdf(
                            f_xctrs, 2, scale=1/(2*Rs_dfv0[q]))
                    fsPDdfv0c1[q] = stats.expon.cdf(
                            T/2, scale = 1/(2*Rs_dfv0[q]))
                    fsPDdfv0c2[q] = stats.erlang.cdf(
                            T/2, 2, scale = 1/(2*Rs_dfv0[q]))
                logging.debug(f'fRGct {fRGct} Rs_dfv0[q] {Rs_dfv0[q]}\n'
                              f'pdfs_Rc_dfv0c1[q] {pdfs_Rc_dfv0c1[q]}\n'
                              f'pdfs_Rc_dfv0c2[q] {pdfs_Rc_dfv0c2[q]}\n'
                              f'cdfs_Rc_dfv0c1[q] {cdfs_Rc_dfv0c1[q]}\n'
                              f'cdfs_Rc_dfv0c2[q] {cdfs_Rc_dfv0c2[q]}\n'
                              f'fsPDdfv0c1[q] {fsPDdfv0c1[q]}\n'
                              f'fsPDdfv0c2[q] {fsPDdfv0c2[q]}')

                # fill dt arrays by subhalo
                for r in range(len(shs_by_m[q])):
                    mrgr_snps_sh = mergers['snap_m'][(mergers['s_id_a'] ==
                                                      shs_by_m[q][r][0])]
                    num_mrgrs = len(mrgr_snps_sh)
                    logging.debug(f's_id_a {shs_by_m[q][r][0]} mrgr_snps_sh '
                                  f'{mrgr_snps_sh} num_mrgrs {num_mrgrs} ')

                    if num_mrgrs == 0:
                        logging.debug('No mergers')
                        dts_c1nc2n[q] += 1
                    else:
                        num_mrgrs_prev = len(mrgr_snps_sh[mrgr_snps_sh <= j])
                        num_mrgrs_next = num_mrgrs - num_mrgrs_prev
                        logging.debug(f'num_mrgrs_prev {num_mrgrs_prev} '
                                      f'num_mrgrs_next {num_mrgrs_next}')

                        # set merger info
                        i_prev, i_next = 0, 0
                        mrgrs = np.zeros(num_mrgrs, dtype=dtype_mrgr_info)
                        mrgrs_prev = np.zeros(num_mrgrs_prev,
                                              dtype=dtype_mrgr_info)
                        mrgrs_next = np.zeros(num_mrgrs_next,
                                              dtype=dtype_mrgr_info)
                        dts_c0n0_inc.fill(0), dts_c1yc2n_inc.fill(0)
                        dts_p1inc.fill(0), dts_p2inc.fill(0)
                        dts_n1inc.fill(0), dts_n2inc.fill(0)

                        for s in range(num_mrgrs):
                            # set times
                            tms = glb.ts[dictnum][mrgr_snps_sh[s]-1]
                            tme = glb.ts[dictnum][mrgr_snps_sh[s]]

                            # set T for this merger
                            if p == 'merger':
                                Tprob = glb.Tsnys[dictnum][mrgr_snps_sh[s]] * u
                            else:
                                Tprob = T
                            logging.debug(
                                    f'Tref {p} snap_m {mrgr_snps_sh[s]} T_m '
                                    f'{glb.Tsnys[dictnum][mrgr_snps_sh[s]]} '
                                    f'Tfac {u} Tprob {Tprob}')

                            # set probabilities
                            if tme <= ta:
                                prob_raw = (Tprob/2 - (ta - tme)) / (tme - tms)
                            else:
                                prob_raw = (Tprob/2 - (tms - ta)) / (tme - tms)
                            if prob_raw > 1:
                                prob = 1
                            elif prob_raw < 0:
                                prob = 0
                            else:
                                prob = prob_raw
                            logging.debug(f'prob_raw {prob_raw}')

                            mrgrs[s] = ((mrgr_snps_sh[s]-1, tms, 
                                         mrgr_snps_sh[s], tme, prob))
                            if mrgr_snps_sh[s] <= j:
                                mrgrs_prev[i_prev] = ((mrgr_snps_sh[s]-1, tms, 
                                                       mrgr_snps_sh[s], tme,
                                                       prob))
                                i_prev += 1
                            else:
                                mrgrs_next[i_next] = ((mrgr_snps_sh[s]-1, tms,
                                                       mrgr_snps_sh[s], tme,
                                                       prob))
                                i_next += 1
                        logging.debug(f'mrgrs {mrgrs}\nmrgrs_prev '
                                      f'{mrgrs_prev}\nmrgrs_next {mrgrs_next}')

                        logging.debug('All mergers')
                        dts_c0n0_inc, dts_c1yc2n_inc = set_dt_incs(dictnum, j,
                                mrgrs, dt_edges, True)
                        logging.debug('Previous mergers')
                        if num_mrgrs_prev > 0:
                            dts_p1inc, dts_p2inc = set_dt_incs(dictnum, j,
                                    mrgrs_prev, dt_edges, False)
                        logging.debug('Next mergers')
                        if num_mrgrs_next > 0:
                            dts_n1inc, dts_n2inc = set_dt_incs(dictnum, j,
                                    mrgrs_next, dt_edges, False)
                        
                        dts_c0n0[q] = dts_c0n0[q] + dts_c0n0_inc
                        dts_c1yc2n[q] = dts_c1yc2n[q] + dts_c1yc2n_inc
                        dts1p[q] = dts1p[q] + dts_p1inc
                        dts2p[q] = dts2p[q] + dts_p2inc
                        dts1n[q] = dts1n[q] + dts_n1inc
                        dts2n[q] = dts2n[q] + dts_n2inc
                    logging.debug(f'dts_c0n0[q] {dts_c0n0[q]}\ndts_c1nc2n[q] '
                                  f'{dts_c1nc2n[q]}\ndts_c1yc2n[q] '
                                  f'{dts_c1yc2n[q]}\ndts1p[q] {dts1p[q]}\n'
                                  f'dts2p[q] {dts2p[q]}\ndts1n[q] {dts1n[q]}\n'
						          f'dts2n[q] {dts2n[q]}')

                # compute valid fractions
                fs_vldc1[q] = ((np.sum(dts_c0n0[q]) + sum(dts_c1yc2n[q]))
                               / len(shs_by_m[q]))
                fs_vldc2[q] = np.sum(dts_c0n0[q]) / len(shs_by_m[q])
                logging.debug(f'fs_vldc1 {fs_vldc1}\nfs_vldc2 {fs_vldc2}')

                # compute probability distribution values
                if fs_vldc1[q] > 0:
                    Rs_dfv1[q] = Rs_dfv0[q] / fs_vldc1[q]
                    # multiply R by 2 in below since Poisson distributions are
                    # based on time to *next* event, and we need time to 
                    # *closest*. Also note c2 values below must use same R as
                    # c1 values, so c2 Rs should be divided by fs_vldc1 as
                    # above, not fs_vldc2 as might be expected
                    if Rs_dfv1[q] > 0:
                        pdfs_Rc_dfv1c1[q] = stats.expon.pdf(
                                f_xctrs, scale=1/(2*Rs_dfv1[q]))
                        pdfs_Rc_dfv1c2[q] = stats.erlang.pdf(
                                f_xctrs, 2, scale=1/(2*Rs_dfv1[q]))
                        cdfs_Rc_dfv1c1[q] = stats.expon.cdf(
                                f_xctrs, scale=1/(2*Rs_dfv1[q]))
                        cdfs_Rc_dfv1c2[q] = stats.erlang.cdf(
                                f_xctrs, 2, scale=1/(2*Rs_dfv1[q]))
                        fsPDdfv1c1[q] = stats.expon.cdf(
                                T/2, scale=1/(2*Rs_dfv1[q]))
                        fsPDdfv1c2[q] = stats.erlang.cdf(
                                T/2, 2, scale=1/(2*Rs_dfv1[q]))
                logging.debug(f'Rs_dfv1 {Rs_dfv1}\n'
                              f'pdfs_Rc_dfv1c1[q] {pdfs_Rc_dfv1c1[q]}\n'
                              f'pdfs_Rc_dfv1c2[q] {pdfs_Rc_dfv1c2[q]}\n'
                              f'cdfs_Rc_dfv1c1[q] {cdfs_Rc_dfv1c1[q]}\n'
                              f'cdfs_Rc_dfv1c2[q] {cdfs_Rc_dfv1c2[q]}\n'
                              f'fsPDdfv1c1[q] {fsPDdfv1c1[q]}\n'
                              f'fsPDdfv1c2[q] {fsPDdfv1c2[q]}')

                # get 1c and 2c sums
                dts_c1_sums = np.sum(dts_c0n0[q], axis=0) + dts_c1yc2n[q]
                dts_c2_sums = np.sum(dts_c0n0[q], axis=1)
                dts_c1_sum = np.sum(dts_c1_sums)
                dts_c2_sum = np.sum(dts_c2_sums)
                logging.debug(f'dts_c1_sums {dts_c1_sums} '
                              f'dts_c2_sums {dts_c2_sums} '
                              f'dts_c1_sum {dts_c1_sum} '
                              f'dts_c2_sum {dts_c2_sum}')

                # populate cumulative/normalized bins
                for r in range(len(dt_ctrs)):
                    # populate normalized, noncumulative bins
                    if dts_c1_sum > 0:
                        dts_c1_c0n1[q][r] = (dts_c1_sums[r] / dts_c1_sum
                                             / dt_wdths[r])
                    if dts_c2_sum > 0:
                        dts_c2_c0n1[q][r] = (dts_c2_sums[r] / dts_c2_sum
                                             / dt_wdths[r])

                    # populate cumulative bins, normalized and non-
                    if r == 0:
                        dts_c1_c1n0[q][r] = dts_c1_sums[r]
                        dts_c1_c1n1[q][r] = dts_c1_c0n1[q][r] * dt_wdths[r]
                        dts_c2_c1n0[q][r] = dts_c2_sums[r]
                        dts_c2_c1n1[q][r] = dts_c2_c0n1[q][r] * dt_wdths[r]
                    else:
                        dts_c1_c1n0[q][r] = (dts_c1_c1n0[q][r-1]
                                             + dts_c1_sums[r])
                        dts_c1_c1n1[q][r] = (dts_c1_c1n1[q][r-1]
                                             + dts_c1_c0n1[q][r] * dt_wdths[r])
                        dts_c2_c1n0[q][r] = (dts_c2_c1n0[q][r-1]
                                             + dts_c2_sums[r])
                        dts_c2_c1n1[q][r] = (dts_c2_c1n1[q][r-1]
                                             + dts_c2_c0n1[q][r] * dt_wdths[r])
                logging.debug(f'dts_c1_c0n1[q] {dts_c1_c0n1[q]}\n'
                              f'dts_c1_c1n0[q] {dts_c1_c1n0[q]}\n'
                              f'dts_c1_c1n1[q] {dts_c1_c1n1[q]}\n'
                              f'dts_c2_c0n1[q] {dts_c2_c0n1[q]}\n'
                              f'dts_c2_c1n0[q] {dts_c2_c1n0[q]}\n'
                              f'dts_c2_c1n1[q] {dts_c2_c1n1[q]}')

                # compute constant-dt distributions
                for r in range(cfg.fxs_num):
                    # compute pdfs
                    if Rs_dfv0[q] > 0:
                        if (f_xctrs[r] >= 0
                            and f_xctrs[r] <= 1/(2 * Rs_dfv0[q])):
                                pdfs_dtc_dfv0c1[q][r] = 2 * Rs_dfv0[q]
                        if (f_xctrs[r] >= 1/(2 * Rs_dfv0[q]) 
                            and f_xctrs[r] <= 1/Rs_dfv0[q]):
                                pdfs_dtc_dfv0c2[q][r] = 2 * Rs_dfv0[q]
                    if Rs_dfv1[q] > 0:
                        if (f_xctrs[r] >= 0
                            and f_xctrs[r] <= 1 / (2 * Rs_dfv1[q])):
                                pdfs_dtc_dfv1c1[q][r] = 2 * Rs_dfv1[q]
                        # pdfs_dtc_dfv1c2 must use same R as pdfs_dtc_dfv1c1,
                        # so its R should be divided by fs_vldc1 as below, not
                        # fs_vldc2 as might be expected
                        if (f_xctrs[r] >= 1 / (2 * Rs_dfv1[q]) 
                            and f_xctrs[r] <= 1 / Rs_dfv1[q]):
                                pdfs_dtc_dfv1c2[q][r] = 2 * Rs_dfv1[q]

                    # compute cdfs
                    if r == 0:
                        cdfs_dtc_dfv0c1[q][r] = (pdfs_dtc_dfv0c1[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv1c1[q][r] = (pdfs_dtc_dfv1c1[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv0c2[q][r] = (pdfs_dtc_dfv0c2[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv1c2[q][r] = (pdfs_dtc_dfv1c2[q][r]
                                                 * f_xwdths[r])
                    else:
                        cdfs_dtc_dfv0c1[q][r] = (cdfs_dtc_dfv0c1[q][r-1]
                                                 + pdfs_dtc_dfv0c1[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv1c1[q][r] = (cdfs_dtc_dfv1c1[q][r-1]
                                                 + pdfs_dtc_dfv1c1[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv0c2[q][r] = (cdfs_dtc_dfv0c2[q][r-1]
                                                 + pdfs_dtc_dfv0c2[q][r]
                                                 * f_xwdths[r])
                        cdfs_dtc_dfv1c2[q][r] = (cdfs_dtc_dfv1c2[q][r-1]
                                                 + pdfs_dtc_dfv1c2[q][r]
                                                 * f_xwdths[r])
                logging.debug(f'pdfs_dtc_dfv0c1[q] {pdfs_dtc_dfv0c1[q]}\n'
                              f'pdfs_dtc_dfv0c2[q] {pdfs_dtc_dfv0c2[q]}\n'
                              f'pdfs_dtc_dfv1c1[q] {pdfs_dtc_dfv1c1[q]}\n'
                              f'pdfs_dtc_dfv1c2[q] {pdfs_dtc_dfv1c2[q]}\n'
                              f'cdfs_dtc_dfv0c1[q] {cdfs_dtc_dfv0c1[q]}\n'
                              f'cdfs_dtc_dfv0c2[q] {cdfs_dtc_dfv0c2[q]}\n'
                              f'cdfs_dtc_dfv1c1[q] {cdfs_dtc_dfv1c1[q]}\n'
                              f'cdfs_dtc_dfv1c2[q] {cdfs_dtc_dfv1c2[q]}')

                # compute KDE inputs
                KDEc1_inp_i, KDEc2_inp_i = 0, 0
                
                KDEc1_inp_lens = np.rint(dts_c1_sums).astype(int)
                KDEc1_inp = np.full(sum(KDEc1_inp_lens) * 2, np.nan)
                KDEc2_inp_lens = np.rint(dts_c2_sums).astype(int)
                KDEc2_inp = np.full(sum(KDEc2_inp_lens) * 2, np.nan)
                logging.debug(f'KDEc1_inp_lens {KDEc1_inp_lens}\n'
                              f'KDEc1_inp {KDEc1_inp}\n'    
                              f'KDEc2_inp_lens {KDEc2_inp_lens}\n'
                              f'KDEc2_inp {KDEc2_inp}')
                
                for r in range(len(dt_ctrs)):
                    # fill KDEc1 input array
                    if KDEc1_inp_lens[r] > 0:
                        KDEc1_inp_wdth = dt_wdths[r] / (KDEc1_inp_lens[r] + 1)
                        KDEc1_inp_crnt = np.linspace(
                                dt_edges[r] + KDEc1_inp_wdth,
                                dt_edges[r+1] - KDEc1_inp_wdth, 
                                KDEc1_inp_lens[r])
                        for s in range(KDEc1_inp_lens[r]):
                            KDEc1_inp[2*s+KDEc1_inp_i] = KDEc1_inp_crnt[s]
                            # negative value included to mirror KDE input
                            # across y-axis, so KDE doesn't start at 0 when
                            # dt = 0
                            KDEc1_inp[2*s+KDEc1_inp_i+1] = -KDEc1_inp_crnt[s]
                        KDEc1_inp_i = KDEc1_inp_i + KDEc1_inp_lens[r] * 2
                        logging.debug(f'KDEc1_inp_wdth {KDEc1_inp_wdth}\n'
                                      f'KDEc1_inp_crnt {KDEc1_inp_crnt}\n'
                                      f'KDEc1_inp {KDEc1_inp}\n'
                                      f'KDEc1_inp_i {KDEc1_inp_i}')

                    # fill KDEc2 input array
                    if KDEc2_inp_lens[r] > 0:
                        KDEc2_inp_wdth = dt_wdths[r] / (KDEc2_inp_lens[r] + 1)
                        KDEc2_inp_crnt = np.linspace(
                                dt_edges[r] + KDEc2_inp_wdth, 
                                dt_edges[r+1] - KDEc2_inp_wdth,
                                KDEc2_inp_lens[r])
                        for s in range(KDEc2_inp_lens[r]):
                            KDEc2_inp[2*s+KDEc2_inp_i] = KDEc2_inp_crnt[s]
                            # negative value included to mirror KDE input
                            # across y-axis, so KDE doesn't start at 0 when
                            # dt = 0
                            KDEc2_inp[2*s+KDEc2_inp_i+1] = -KDEc2_inp_crnt[s]
                        KDEc2_inp_i = KDEc2_inp_i + KDEc2_inp_lens[r] * 2
                        logging.debug(f'KDEc2_inp_wdth {KDEc2_inp_wdth}\n'
                                      f'KDEc2_inp_crnt {KDEc2_inp_crnt}\n'
                                      f'KDEc2_inp {KDEc2_inp}\n'
                                      f'KDEc2_inp_i {KDEc2_inp_i}')    

                # check for unfilled input arrays
                if np.isnan(np.sum(KDEc1_inp)):
                    raise ValueError('KDEc1_inp not full')
                if np.isnan(np.sum(KDEc2_inp)):
                    raise ValueError('KDEc2_inp not full')
                
                # compute first-closest KDEs
                vals_same = True
                if len(KDEc1_inp) >= 1:
                    for r in range(len(KDEc1_inp)):
                        if abs(KDEc1_inp[r]) != KDEc1_inp[0]:
                            vals_same = False
                            break
                    if vals_same:
                        pdfsKDEc1[q] = stats.norm.pdf(f_xctrs,
                                                      loc=KDEc1_inp[0])
                        fsKDEc1[q] = stats.norm(KDEc1_inp[0]).cdf(T/2)
                    else:
                        KDEc1_krnl = stats.gaussian_kde(KDEc1_inp)
                        # "* 2" in next and following line to compensate for
                        # KDE mirroring across y-axis
                        pdfsKDEc1[q] = KDEc1_krnl.evaluate(f_xctrs) * 2
                        fsKDEc1[q] = KDEc1_krnl.integrate_box_1d(0, T/2) * 2
                logging.debug(f'pdfsKDEc1[q]\n{pdfsKDEc1[q]}\n'
                              f'fsKDEc1[q] {fsKDEc1[q]}')

                # # compute second-closest KDEs
                vals_same = True
                if len(KDEc2_inp) >= 1:
                    for r in range(len(KDEc2_inp)):
                        if abs(KDEc2_inp[r]) != KDEc2_inp[0]:
                            vals_same = False
                            break
                    if vals_same:
                        pdfsKDEc2[q] = stats.norm.pdf(f_xctrs,
                                                      loc=KDEc2_inp[0])
                        fsKDEc2[q] = stats.norm(KDEc2_inp[0]).cdf(T/2)
                    else:
                        KDEc2_krnl = stats.gaussian_kde(KDEc2_inp)
                        # "* 2" in next and following line to compensate for
                        # KDE mirroring across y-axis
                        pdfsKDEc2[q] = KDEc2_krnl.evaluate(f_xctrs) * 2
                        fsKDEc2[q] = KDEc2_krnl.integrate_box_1d(0, T/2) * 2
                logging.debug(f'pdfsKDEc2[q] {pdfsKDEc2[q]}\n'
                              f'fsKDEc2[q] {fsKDEc2[q]}')
                
            # save results
            np.savez_compressed(
                    os.path.join(pathname, filename),
                    Rs_dfv0=Rs_dfv0, fs_vldc1=fs_vldc1, fs_vldc2=fs_vldc2,
                    fsKDEc1=fsKDEc1, fsKDEc2=fsKDEc2,
                    fsPDdfv0c1=fsPDdfv0c1, fsPDdfv0c2=fsPDdfv0c2, 
                    fsPDdfv1c1=fsPDdfv1c1, fsPDdfv1c2=fsPDdfv1c2,
                    dts_c0n0=dts_c0n0,
                    dts_c1yc2n=dts_c1yc2n, dts_c1nc2n = dts_c1nc2n,
                    dts_c1_c0n1=dts_c1_c0n1, dts_c1_c1n0=dts_c1_c1n0,
                    dts_c1_c1n1=dts_c1_c1n1, dts_c2_c0n1=dts_c2_c0n1,
                    dts_c2_c1n0=dts_c2_c1n0, dts_c2_c1n1=dts_c2_c1n1, 
                    dts1p=dts1p, dts2p=dts2p, dts1n=dts1n, dts2n=dts2n,
                    pdfs_dtc_dfv0c1=pdfs_dtc_dfv0c1, 
                    pdfs_dtc_dfv0c2=pdfs_dtc_dfv0c2,
                    pdfs_dtc_dfv1c1=pdfs_dtc_dfv1c1,
                    pdfs_dtc_dfv1c2=pdfs_dtc_dfv1c2,
                    pdfs_Rc_dfv0c1=pdfs_Rc_dfv0c1,
                    pdfs_Rc_dfv0c2=pdfs_Rc_dfv0c2,
                    pdfs_Rc_dfv1c1=pdfs_Rc_dfv1c1,
                    pdfs_Rc_dfv1c2=pdfs_Rc_dfv1c2,
                    pdfsKDEc1=pdfsKDEc1, pdfsKDEc2=pdfsKDEc2,
                    cdfs_dtc_dfv0c1=cdfs_dtc_dfv0c1,
                    cdfs_dtc_dfv0c2=cdfs_dtc_dfv0c2, 
                    cdfs_dtc_dfv1c1=cdfs_dtc_dfv1c1,
                    cdfs_dtc_dfv1c2=cdfs_dtc_dfv1c2,
                    cdfs_Rc_dfv0c1=cdfs_Rc_dfv0c1,
                    cdfs_Rc_dfv0c2=cdfs_Rc_dfv0c2,
                    cdfs_Rc_dfv1c1=cdfs_Rc_dfv1c1,
                    cdfs_Rc_dfv1c2=cdfs_Rc_dfv1c2)

def viewdtdata():
    """
    View dt-related raw output data.
    """

    print('Running viewdtdata')

    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        _, _, ilrun, _, _, _ = get_run_info(ilnum)
        
        for j, k, n, o, p, u in it.product(snapnums, cfg.mu_maxes,
                                           cfg.virtualprogs, cfg.SubLink_gals,
                                           cfg.Trefs, cfg.Tfacs):
            fcfg_dt = (f'i{ilnum}s{j}r{k}v{n}g{o}Tr{p[0]}Tf{u}'
                       f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                       f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
                       f'Rn{cfg.RGms_num}do{cfg.dtbinwdthopt:1.1f}'
                       f'fn{cfg.fxs_num}Km{cfg.KDEmult}ss{cfg.subhalostart}'
                       f'se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'dt', ilrun,
                                      'dt_dat' + fcfg_dt + '.npz')) as data:
                with open('dt_dat' + fcfg_dt + '.txt', 'w') as f:
                    for i in data:
                        f.write(i + '\n')
                        for j in range(len(data[i])):
                            f.write(str(data[i][j]) + '\n')

def setfmin():
    """Determines minimum merger fraction.

    Parameters
    --------
    None

    Returns
    --------
    None
    """
    print(f'Running setfmin subhalostart {cfg.subhalostart} '
          f'subhalo_end {cfg.subhalo_end}')

    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    
    # get fmin from setfs-derived values
    fcfg_f = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
              f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
              f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
              f'Rn{cfg.RGms_num}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with np.load(os.path.join('output', 'numerical', 'f',
                              'fdat' + fcfg_f + '.npz')) as data:
        fs_fcn_all = data['fs_fcn_all']
        fs_ct_all = data['fs_ct_all']
    logging.debug(f'fs_fcn_all\n{fs_fcn_all}\nfs_ct_all\n{fs_ct_all}')

    fsRGfcnmin, fsRGctmin, fsbinmin, fsmltmin = 1, 1, 1, 1
    fs1min, fs2min, fs3min, fs4min, = 1, 1, 1, 1
    if len(fs_fcn_all['fsRGfcn'][fs_fcn_all['fsRGfcn'] > 0]) > 0:
        fsRGfcnmin = np.amin(fs_fcn_all['fsRGfcn'][fs_fcn_all['fsRGfcn'] > 0])
    if len(fs_ct_all['fsbin'][fs_ct_all['fsbin'] > 0]) > 0:
        fsbinmin = np.amin(fs_ct_all['fsbin'][fs_ct_all['fsbin'] > 0])
    if len(fs_ct_all['fsmlt'][fs_ct_all['fsmlt'] > 0]) > 0:
        fsmltmin = np.amin(fs_ct_all['fsmlt'][fs_ct_all['fsmlt'] > 0])
    if len(fs_ct_all['fsRGct'][fs_ct_all['fsRGct'] > 0]) > 0:
        fsRGctmin = np.amin(fs_ct_all['fsRGct'][fs_ct_all['fsRGct'] > 0])
    if len(fs_ct_all['fs1'][fs_ct_all['fs1'] > 0]) > 0:
        fs1min = np.amin(fs_ct_all['fs1'][fs_ct_all['fs1'] > 0])
    if len(fs_ct_all['fs2'][fs_ct_all['fs2'] > 0]) > 0:
        fs2min = np.amin(fs_ct_all['fs2'][fs_ct_all['fs2'] > 0])
    if len(fs_ct_all['fs3'][fs_ct_all['fs3'] > 0]) > 0:
        fs3min = np.amin(fs_ct_all['fs3'][fs_ct_all['fs3'] > 0])
    if len(fs_ct_all['fs4'][fs_ct_all['fs4'] > 0]) > 0:
        fs4min = np.amin(fs_ct_all['fs4'][fs_ct_all['fs4'] > 0])
    logging.debug(f'fsRGfcnmin {fsRGfcnmin} fsbinmin {fsbinmin} '
                  f'fsmltmin {fsmltmin}\nfs1min {fs1min} fs2min {fs2min} '
                  f'fs3min {fs3min} fs4min {fs4min} fsRGctmin {fsRGctmin} ')

    # get dt-derived fs
    fsKDEmvldc1min, fsKDEmvldc2min = 1, 1
    fsPDdfv0c1min, fsPDdfv0c2min = 1, 1
    fsPDdfv1c1min, fsPDdfv1c2min = 1, 1
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        _, _, ilrun, _, _, _ = get_run_info(ilnum)
        
        for j, k, n, o, p, u in it.product(snapnums, cfg.mu_maxes,
                                           cfg.virtualprogs, cfg.SubLink_gals,
                                           cfg.Trefs, cfg.Tfacs):
            fcfg_dt = (f'i{ilnum}s{j}r{k}v{n}g{o}Tr{p[0]}Tf{u}'
                       f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                       f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
                       f'Rn{cfg.RGms_num}do{cfg.dtbinwdthopt:1.1f}'
                       f'fn{cfg.fxs_num}Km{cfg.KDEmult}ss{cfg.subhalostart}'
                       f'se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'dt', ilrun,
                                      'dt_dat' + fcfg_dt + '.npz')) as data:
                fs_vldc1 = data['fs_vldc1']
                fs_vldc2 = data['fs_vldc2']
                fsKDEc1 = data['fsKDEc1']
                fsKDEc2 = data['fsKDEc2']
                fsPDdfv0c1 = data['fsPDdfv0c1']
                fsPDdfv0c2 = data['fsPDdfv0c2']
                fsPDdfv1c1 = data['fsPDdfv1c1']
                fsPDdfv1c2 = data['fsPDdfv1c2']
            logging.debug(f'fs_vldc1 {fs_vldc1}\nfs_vldc2 {fs_vldc2}\n'
                          f'fsKDEc1 {fsKDEc1}\nfsKDEc2 {fsKDEc2}\n'
                          f'fsPDdfv0c1 {fsPDdfv0c1}\n'
                          f'fsPDdfv0c2 {fsPDdfv0c2}\n'
                          f'fsPDdfv1c1 {fsPDdfv1c1}\n'
                          f'fsPDdfv1c2 {fsPDdfv1c2}')

            for r in range(len(fsKDEc1)):
                if fsKDEc1[r] >= 0 and fs_vldc1[r] >= 0:
                    fsKDEmvldc1 = fsKDEc1[r] * fs_vldc1[r]
                    if fsKDEmvldc1 > 0 and fsKDEmvldc1 < fsKDEmvldc1min:
                        fsKDEmvldc1min = fsKDEmvldc1
                
                if fsKDEc2[r] >= 0 and fs_vldc2[r] >= 0:
                    fsKDEmvldc2 = fsKDEc2[r] * fs_vldc2[r]
                    if fsKDEmvldc2 > 0 and fsKDEmvldc2 < fsKDEmvldc2min:
                        fsKDEmvldc2min = fsKDEmvldc2
                logging.debug(f'fsKDEc1[r] {fsKDEc1[r]} fs_vldc1[r] '
                              f'{fs_vldc1[r]} fsKDEmvldc1min '
                              f'{fsKDEmvldc1min}\nfsKDEc2[r] {fsKDEc2[r]} '
                              f'fs_vldc2[r] {fs_vldc2[r]} fsKDEmvldc2min '
                              f'{fsKDEmvldc2min}')
            if len(fsPDdfv0c1[fsPDdfv0c1 > 0]):
                fsPDdfv0c1min = np.amin(fsPDdfv0c1[fsPDdfv0c1 > 0])
            if len(fsPDdfv0c2[fsPDdfv0c2 > 0]):
                fsPDdfv0c2min = np.amin(fsPDdfv0c2[fsPDdfv0c2 > 0])
            if len(fsPDdfv1c1[fsPDdfv1c1 > 0]):
                fsPDdfv1c1min = np.amin(fsPDdfv1c1[fsPDdfv1c1 > 0])
            if len(fsPDdfv1c2[fsPDdfv1c2 > 0]):
                fsPDdfv1c2min = np.amin(fsPDdfv1c2[fsPDdfv1c2 > 0])
            logging.debug(f'fsKDEmvldc1min {fsKDEmvldc1min}\n'
                          f'fsKDEmvldc2min {fsKDEmvldc2min}\n'
                          f'fsPDdfv0c1min {fsPDdfv0c1min}\n'
                          f'fsPDdfv0c2min {fsPDdfv0c2min}\n'
                          f'fsPDdfv1c1min {fsPDdfv1c1min}\n'
                          f'fsPDdfv1c2min {fsPDdfv1c2min}')

    fmin = min(fsRGfcnmin, fsbinmin, fsmltmin, fs1min, fs2min, fs3min, fs4min,
               fsRGctmin, fsKDEmvldc1min, fsKDEmvldc2min, fsPDdfv0c1min,
               fsPDdfv0c2min, fsPDdfv1c1min,fsPDdfv1c2min)
    if fmin < cfg.fminmin:
        fmin = cfg.fminmin

    logging.debug(f'fmin {fmin}')
    print(f'fmin {fmin}')

    # display/store fmin
    pathname = os.path.join('output', 'numerical', 'fmin')
    if (not os.path.exists(pathname)):
        os.makedirs(pathname)
    fcfg_fmin = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
                 f'Rn{cfg.RGms_num}do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}'
                 f'Km{cfg.KDEmult}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join(pathname, 'fmin' + fcfg_fmin + '.txt'), 'w') as f:
        f.write(str(fmin))

def set_errbnds(num_mrgrs, numgxy):
    """
    Determines the lower and upper error bounds of a value using a
    Wilson score interval with continuity correction.

    Parameters
    ----
    num_mrgrs : float
        The equivalent number of mergers.
    numgxy: float
        The number of subhalos.

    Returns
    --------
    tuple : (float, float)
        Tuple of lower and upper bounds of error bars.
    """
    
    z = 1
    leftnum = 2 * num_mrgrs + z**2
    left_rdcnd = z**2 - 1 / numgxy + 4 * num_mrgrs * (1 - num_mrgrs / numgxy)
    rt_rdcnd = 4 * num_mrgrs / numgxy - 2
    denom = 2 * (numgxy + z**2)
    logging.debug(
            f'num_mrgrs {num_mrgrs} numgxy {numgxy} leftnum {leftnum} '
            f'left_rdcnd {left_rdcnd} rt_rdcnd {rt_rdcnd} denom {denom} ')
    
    # set lower bound
    if num_mrgrs == 0:
        lwrbnd = 0
    else:
        if left_rdcnd + rt_rdcnd < 0:
            lwrbnd_rdcnd = 0
            logging.debug('Lower bound radicand < 0')
        else:
            lwrbnd_rdcnd = left_rdcnd + rt_rdcnd
        lwrbnd = max(0, (leftnum - (z * math.sqrt(lwrbnd_rdcnd) + 1)) / denom)
        
    # set upper bound
    if num_mrgrs == numgxy:
        uprbnd = 1
    else:
        if left_rdcnd - rt_rdcnd < 0:
            uprbnd_rdcnd = 0
            logging.debug('Upper bound radicand < 0')
        else:
            uprbnd_rdcnd = left_rdcnd - rt_rdcnd
        uprbnd = min(1, (leftnum + (z * math.sqrt(uprbnd_rdcnd) + 1)) / denom)
    
    logging.debug(f'lwrbnd {lwrbnd} uprbnd {uprbnd}')
    return (lwrbnd, uprbnd)

def createfvmplot(ilnum, snapnum, mu_min, mu_max, virtualprog, SubLink_gal,
                  Tref, Tfac, m_axis_max, fmin, RGm_edges, fsRGfcn, m_ctrs,
                  numgxy, fdat, subset):
    """
    Create an f vs m plot.

    Parameters
    ----
    ilnum: int
        Illustris run number.
    snapnum : int
        Analysis snapshot number.
    mu_min : float
        Minimum acceptable valid merger mass ratio.
    mu_max : float
        Maximum acceptable valid merger mass ratio.
    virtualprog : boolean
        True if virtual progenitors created, false if not.
    SubLink_gal : boolean
        True if SubLink_gal trees used, false if SubLInk trees used.
    Tref : string
        The reference used to determine T.
    Tfac : float
        The factor by which T is multiplied.
    m_axis_max : float
        Maximum value of plot mass axis.
    fmin : float
        Minimum merger fraction.
    RGm_edges : ndarray [float]
        Input values to R-G function.
    fsRGfcn : ndarray [float]
        Output values of R-G function.
    m_ctrs : ndarray [float]
        Mass bin centers of data to be plotted.
    numgxy : ndarray [float]
        Number of galaxies in each mass bin.
    fdat : ndarray [float]
        Data to be plotted.
    subset : string
        Identifier for subset of values to be plotted

    Returns
    --------
    None
    """

    print(f'Creating f vs m plot at ilnum {ilnum} snapnum {snapnum} '
          f'mu_min {mu_min} mu_max {mu_max} virtualprog {virtualprog} '
          f'SubLink_gal {SubLink_gal} Tref {Tref} Tfac {Tfac} '
          f'subhalostart {cfg.subhalostart} subhalo_end {cfg.subhalo_end}')

    dictnum, _, ilrun, _, _, _ = get_run_info(ilnum)
    numbins = len(m_ctrs)
    numlines = len(fdat)
    if 'vldc1' in fdat['name']:
        numlines = numlines - 1
    if 'vldc2' in fdat['name']:
        numlines = numlines - 1
    logging.debug(f'numlines {numlines}')
    
    # calculate error bars    
    if cfg.ploterrorbars != 0:
        # fill error array
        dtype_errs = [('name', np.unicode_, 10),
                      ('num_mgrs', np.float_, numbins), 
                      ('e_abs_lo', np.float_, numbins),
                      ('e_abs_hi', np.float_, numbins),
                      ('e_rel_lo', np.float_, numbins),
                      ('e_rel_hi', np.float_, numbins)]
        errs = np.zeros(numlines, dtype=dtype_errs)
    
        # get ta, tam1, T
        ta = glb.ts[dictnum][snapnum]
        tam1 = glb.ts[dictnum][snapnum-1]
        if Tref == 'analysis' or Tref == 'merger':
            T = glb.Tsnys[dictnum][snapnum] * Tfac
        elif Tref == 'snapwidth':
            T = glb.Tsnps[dictnum][snapnum] * Tfac
        logging.debug(f'Tref {Tref} ta {ta} tam1 {tam1} Tfac {Tfac} T {T}')
        
        # set errors
        for i in range(numlines):
            errs[i]['name'] = fdat[i]['name']
            for j in range(numbins):
                if (numgxy[j] > 0 and fdat[i]['data'][j] > 0
                    and fdat[i]['data'][j] <= 1):
                    if fdat[i]['name'] == 'RGct':
                        num_mgrs = (fdat[i]['data'][j] * numgxy[j] 
                                    * (ta - tam1) / T)
                    else:
                        num_mgrs = fdat['data'][i][j] * numgxy[j]                        
                    
                    e_abs_lo, e_abs_hi = set_errbnds(num_mgrs, numgxy[j])

                    if fdat[i]['name'] == 'RGct':
                        e_rel_lo = (fdat[i]['data'][j]
                                    - e_abs_lo / (ta - tam1) * T)
                        e_rel_hi = (e_abs_hi / (ta - tam1) * T
                                    - fdat[i]['data'][j])
                    elif fdat[i]['name'] == 'KDEc1':
                        logging.debug(
                            "vldc1"
                            f"{fdat[fdat['name'] == 'vldc1']['data'][0][j]}")
                        e_rel_lo = (
                                fdat[fdat['name'] == 'vldc1']['data'][0][j]
                                * (fdat[i]['data'][j] - e_abs_lo))
                        e_rel_hi = (
                                fdat[fdat['name'] == 'vldc1']['data'][0][j] 
                                * (e_abs_hi - fdat[i]['data'][j]))
                    elif fdat[i]['name'] == 'KDEc2':
                        logging.debug(
                            "vldc2 "
                            f"{fdat[fdat['name'] == 'vldc2']['data'][0][j]}")
                        e_rel_lo = (
                                fdat[fdat['name'] == 'vldc2']['data'][0][j]
                                * (fdat[i]['data'][j] - e_abs_lo))
                        e_rel_hi = (
                                fdat[fdat['name'] == 'vldc2']['data'][0][j] 
                                * (e_abs_hi - fdat[i]['data'][j]))
                    else:
                        e_rel_lo = fdat[i]['data'][j] - e_abs_lo
                        e_rel_hi = e_abs_hi - fdat[i]['data'][j]
                        
                    errs[i]['num_mgrs'][j] = num_mgrs
                    errs[i]['e_abs_lo'][j] = e_abs_lo
                    errs[i]['e_abs_hi'][j] = e_abs_hi
                    errs[i]['e_rel_lo'][j] = e_rel_lo      
                    errs[i]['e_rel_hi'][j] = e_rel_hi
                logging.debug(f'i {i} j {j}\nerrs {errs}')
                        
        # address nonpositive relative errors
        logging.debug(f'errs, before relative 0 removal\n{errs}')
        for i in range(numlines):
            for j in range(numbins):
                if errs[i]['e_rel_lo'][j] < 0:
                    warnings.warn(f"e_rel_lo < 0: {errs[i]['name']} mbin {j}, "
                                  'setting to np.nan')
                if errs[i]['e_rel_hi'][j] < 0:
                    warnings.warn(f"e_rel_hi < 0: {errs[i]['name']} mbin {j}, "
                                  'setting to np.nan')
                if errs[i]['e_rel_lo'][j] <= 0:
                    errs[i]['e_rel_lo'][j] = np.nan
                if errs[i]['e_rel_hi'][j] <= 0:
                    errs[i]['e_rel_hi'][j] = np.nan
        logging.debug(f'errs, after relative 0s converted to NaN\n{errs}')
            
    # convert 0 and -1 data (non-error) values to NaN
    for i in range(numlines):
        for j in range(numbins):
            if fdat[i]['data'][j] <= 0:
                fdat[i]['data'][j] = np.nan
    logging.debug(f'fdat, after 0s and -1s converted to NaN\n{fdat}')
  
    namedict = {'RGct': 'Total, RG15', 'bin': 'Total, this work', 
                'mlt': 'Multiple, this work', 'KDEc1': r'KDE*$f_v$, 1c',
                'KDEc2': r'KDE*$f_v$, 2c',
                'PDdfv0c1': '$\lambda=2R$, total', 
                'PDdfv1c1': '$\lambda=2R/F$, total',
                'PDdfv0c2': '$\lambda=2R$, multiple',
                'PDdfv1c2': '$\lambda=2R/F$, multiple',
                '1': 'Binary', '2': 'Trinary',
                '3': 'Quaternary', '4': 'Quinary'}
    colordict = {'RGct': 'c', 'bin': 'g', 'mlt': 'r',
                 'KDEc1': 'm', 'KDEc2': 'y',
                 'PDdfv0c1': 'C0', 'PDdfv1c1': 'C1',
                 'PDdfv0c2': 'C4', 'PDdfv1c2': 'C5',
                 '1': 'C6', '2': 'C7', '3': 'C8', '4': 'C9'}
    markerdict = {'RGct': 'p', 'bin': 's', 'mlt': 'o', 'KDEc1': 'X',
                  'KDEc2': 'D', 'PDdfv0c1': 'v', 'PDdfv1c1': '^',
                  'PDdfv0c2': '<', 'PDdfv1c2': '>', '1': '1', '2': '2',
                  '3': '3', '4': '4'} 
    
    # plot fractions
    if subset == 'a':
        plt.plot(RGm_edges*10**10, fsRGfcn, color='b', label='Total, RG15 fit')
    for i in range(numlines):

        # don't plot /fv F values in publication plots
        if (cfg.functions[cfg.fnum] == 'createpubplots'
                and 'dfv0' in fdat[i]['name']):
            continue
        
        # change fit markers in publication plots
        if (cfg.functions[cfg.fnum] == 'createpubplots'
                and 'dfv1' in fdat[i]['name']):
            line_alpha = 0.5
            markerfill = 'none'
            mrkrsize = 12
            if fdat[i]['name'] == 'PDdfv1c1':
                linecolor = 'g'
                markershape = 's'
            elif fdat[i]['name'] == 'PDdfv1c2':
                linecolor = 'r'
                markershape = 'o'
        else:
            linecolor = colordict[fdat[i]['name']]
            line_alpha = 1
            markershape = markerdict[fdat[i]['name']]
            markerfill = colordict[fdat[i]['name']]
            mrkrsize = 6
        
        # set values
        if fdat[i]['name'] == 'KDEc1':
            ydat = (fdat[i]['data'] * fdat[fdat['name']=='vldc1']['data'][0])
        elif fdat[i]['name'] == 'KDEc2':
            ydat = (fdat[i]['data'] * fdat[fdat['name']=='vldc2']['data'][0])
        else:
            ydat = fdat[i]['data']
 
        # plot points
        if cfg.ploterrorbars == 0:
            plt.plot(m_ctrs*10**10, ydat, alpha=line_alpha,
                     color=linecolor, label=namedict[fdat[i]['name']],
                     marker=markershape, markersize=mrkrsize,
                     markerfacecolor=markerfill, ls='None')
        else:
            if all(np.isnan(fdat[i]['data'])):
                yerrs = None
            else:
                yerrs = (errs[i]['e_rel_lo'], errs[i]['e_rel_hi'])
                
            plt.errorbar(m_ctrs*10**10, ydat, yerr=yerrs,
                         label=namedict[fdat[i]['name']],
                         color=linecolor, alpha=line_alpha, marker=markershape,
                         markersize=mrkrsize, markerfacecolor=markerfill,
                         ls='None')

    # plot legend/titles/etc 
    if cfg.functions[cfg.fnum] == 'createpubplots':
        plt.legend(bbox_to_anchor=(0.99, 0.01), loc='lower right',
                   borderaxespad=0.0, prop={'size':12}, handletextpad=0.2,
                   labelspacing=0)    
    else:
        suptxt = 'Merger Fraction vs Mass'
        if subset == 'a':
            suptxt = suptxt + ', as Measured'
            if platform.node().startswith('jupyter'):
                if cfg.plot_fKDE == True:
                    figtxt_lcn = 0.85
                else:
                    figtxt_lcn = 0.86
            else:
                if cfg.plot_fKDE == True:
                    figtxt_lcn = 0.84
                else:
                    figtxt_lcn = 0.85
        if subset == 'b':
            suptxt = suptxt + ', $p$ Densities'
            if platform.node().startswith('jupyter'):
                figtxt_lcn = 0.82
            else:
                figtxt_lcn = 0.81
        if subset == 'c':
            suptxt = suptxt + ", '-ary'"
            if platform.node().startswith('jupyter'):
                figtxt_lcn = 0.86
            else:
                figtxt_lcn = 0.85
                
        plt.suptitle(suptxt, fontsize=14)
        plt.title(f'{ilrun}, $z$ = {glb.zs[dictnum][snapnum]:.1f}, '
                  f'$\mu$ = {mu_min}-{mu_max}')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                   borderaxespad=0.0, prop={'size':8}, handletextpad=0.2,
                    labelspacing=0)    
        figtxt = (f'Virtual Progs: {str(bool(virtualprog))[0]}\n'
                  f'SubLink_gal: {str(bool(SubLink_gal))[0]}\n'
                  f'$T_{{ref}}$: {Tref}\n$T_{{fac}}$: {Tfac}\n'
                  f'# Gals: {np.sum(numgxy)}')
        plt.figtext(figtxt_lcn, 0.15, figtxt, size=8, linespacing=1)
    
    plt.xlabel(r'$M_{\ast}$ [$M_{sun}$]')
    plt.ylabel('Merger fraction')
    if cfg.mlogspace:
        plt.xscale('log')
    plt.yscale('log')
    plt.xlim(cfg.mmin*10**10, m_axis_max*10**10)
    plt.ylim(fmin * 0.9, 1)
    mpl.rcParams['errorbar.capsize'] = 4
    
    plt.tight_layout(pad=0.2)
    
    if cfg.plot_tofile:
        if cfg.functions[cfg.fnum] == 'createpubplots':
            pathname = os.path.join('output', 'graphical', 'pub')
        else:
            pathname = os.path.join('output', 'graphical', 'f_vs_m', ilrun)
        if (not os.path.exists(pathname)):
            os.makedirs(pathname)
        plt.savefig(os.path.join(pathname,
                    f'fsn{snapnum:03d}rl{mu_min:1.2f}ru{mu_max:02d}'
                    f'vp{virtualprog:01d}sg{SubLink_gal:01d}Tr{Tref[0]}'
                    f'Tf{Tfac:0.1f}mb{cfg.mbinsnumraw}mlt{cfg.mmrglst3}'
                    f'ss{cfg.subhalostart}se{cfg.subhalo_end}{subset}.png'))
    if cfg.plot_toconsole:
        plt.show()
    plt.clf()
    plt.close()

def createfvmplots():
    """
    Create multiple f vs m plots.
    """
    print('Running createfvmplots')

    logging.debug(f'ilnums {cfg.ilnums} snapnumsOG {cfg.snapnumsOG} '
                  f'snapnumsTNG {cfg.snapnumsTNG} ratios {cfg.mu_maxes} '
                  f'virtualprogs {cfg.virtualprogs} '
                  f'SubLink_gals {cfg.SubLink_gals} Trs {cfg.Trefs} '
                  f'Tfs {cfg.Tfacs} mmin {cfg.mmin} mminvirt {cfg.mminvirt} '
                  f'mbinsnumraw {cfg.mbinsnumraw} mmrglst3 {cfg.mmrglst3} '
                  f'mlogspace {cfg.mlogspace} dtbinwdthopt {cfg.dtbinwdthopt} '
                  f'subhalostart {cfg.subhalostart} '
                  f'subhalo_end {cfg.subhalo_end} '
                  f'mu_maxes_to_plot {cfg.mu_maxes_to_plot} '
                  f'virtualprogs_to_plot {cfg.virtualprogs_to_plot} '
                  f'SubLink_gals_to_plot {cfg.SubLink_gals_to_plot} '
                  f'Trefs_to_plot {cfg.Trefs_to_plot} '
                  f'Tfacs_to_plot {cfg.Tfacs_to_plot}')
   
    # get config strings
    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))

    # set mass-related values
    logging.debug('M-related values')
    fcfg_mmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'mmax',
                           'massmax' + fcfg_mmax + '.txt')) as f:
        massmax = float(f.read())
        
    logging.debug('mass bins')
    m_edges, m_ctrs, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                    cfg.mbinsnumraw, cfg.mlogspace,
                                    cfg.mmrglst3)
    mbinsnum = len(m_ctrs)
    
    logging.debug('R-G edges')
    RGm_edges, _, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                 cfg.RGms_num, cfg.mlogspace, 0)
    if cfg.m_axis_maxmanual == 1:
        with open(os.path.join('output', 'numerical',
                               'm_axis_maxmanual.txt')) as f:
            m_axis_max = float(f.read())
    else:
        m_axis_max = massmax + cfg.bin_pdng
    logging.debug(f'massmax {massmax}\nm_edges {m_edges}\nm_ctrs {m_ctrs}\n'
                  f'm_axis_max {m_axis_max}')   

    # get f min
    fcfg_fmin = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
                 f'Rn{cfg.RGms_num}do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}'
                 f'Km{cfg.KDEmult}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'fmin',
                           'fmin' + fcfg_fmin + '.txt')) as f:
        fmin = float(f.read())
    logging.debug(f'fmin {fmin}')

    # load non-dt derived f data
    fcfg_f = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
              f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
              f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}ml{cfg.mlogspace}'
              f'Rn{cfg.RGms_num}ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with np.load(os.path.join('output', 'numerical', 'f',
                              'fdat' + fcfg_f + '.npz')) as data:
        fs_fcn_all = data['fs_fcn_all']
        fs_ct_all = data['fs_ct_all']
    logging.debug(f'fs_fcn_all\n{fs_fcn_all.dtype.names}\n{fs_fcn_all}\n'
                  f'fs_ct_all\n{fs_ct_all.dtype.names}\n{fs_ct_all}')

    dtype_fdat = [('name', np.unicode_, 10), ('data', np.float_, mbinsnum)]

    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        _, basepath, ilrun, snapnummax, _, numsubgroups = get_run_info(ilnum)

        for snapnum, mu_max, Tref, Tfac in it.product(
                snapnums, cfg.mu_maxes_to_plot, cfg.Trefs_to_plot,
                cfg.Tfacs_to_plot):
            logging.debug(f'ilnum {ilnum} snapnum {snapnum} '
                          f'mu_max {mu_max} Tref {Tref} Tfac {Tfac}')
           
            mu_min = 1 / mu_max

            # get non-virtual-progenitor/non-SubLink_gal dependent data
            fsRGfcn = fs_fcn_all[(fs_fcn_all['ilnum'] == ilnum)
                                 & (fs_fcn_all['snapnum'] == snapnum)
                                 & (fs_fcn_all['mu_min'] == mu_min)
                                 & (fs_fcn_all['mu_max'] == mu_max)
                                 & (fs_fcn_all['Tref'] == Tref)
                                 & (fs_fcn_all['Tfac'] == Tfac)]['fsRGfcn'][0]
            logging.debug(f'fsRGfcn\n{fsRGfcn}')

            for virtualprog, SubLink_gal in it.product(
                    cfg.virtualprogs_to_plot, cfg.SubLink_gals_to_plot):
                logging.debug(f'virtualprog {virtualprog} '
                              f'SubLink_gal {SubLink_gal}')

                # get non-dt-derived f data
                fs_ct = fs_ct_all[(fs_ct_all['ilnum'] == ilnum)
                                  & (fs_ct_all['snapnum'] == snapnum)
                                  & (fs_ct_all['mu_min'] == mu_min)
                                  & (fs_ct_all['mu_max'] == mu_max)
                                  & (fs_ct_all['virtualprog'] == virtualprog)
                                  & (fs_ct_all['SubLink_gal'] == SubLink_gal)
                                  & (fs_ct_all['Tref'] == Tref)
                                  & (fs_ct_all['Tfac'] == Tfac)]
                logging.debug(f'fs_ct\n{fs_ct}')
                numgxy = fs_ct['numgxy'][0]
                logging.debug(f'numgxy\n{numgxy}')

                # get dt-derived f data
                fcfg_dt = (f'i{ilnum}s{snapnum}r{mu_max}v{virtualprog}'
                           f'g{SubLink_gal}Tr{Tref[0]}Tf{Tfac}'
                           f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                           f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}'
                           f'ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                           f'do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}'
                           f'Km{cfg.KDEmult}ss{cfg.subhalostart}'
                           f'se{cfg.subhalo_end}')
                with np.load(os.path.join('output', 'numerical', 'dt', ilrun,
                                          'dt_dat' + fcfg_dt + '.npz'))\
                        as data:
                    fs_vldc1 = data['fs_vldc1']
                    fs_vldc2 = data['fs_vldc2']
                    fsKDEc1 = data['fsKDEc1']
                    fsKDEc2 = data['fsKDEc2']
                    fsPDdfv0c1 = data['fsPDdfv0c1']
                    fsPDdfv0c2 = data['fsPDdfv0c2']
                    fsPDdfv1c1 = data['fsPDdfv1c1']
                    fsPDdfv1c2 = data['fsPDdfv1c2']
                logging.debug(f'fs_vldc1 {fs_vldc1}\nfs_vldc2 {fs_vldc2}\n'
                              f'fsKDEc1 {fsKDEc1}\nfsKDEc2 {fsKDEc2}\n'
                              f'fsPDdfv0c1 {fsPDdfv0c1}\nfsPDdfv0c2 '
                              f'{fsPDdfv0c2}\nfsPDdfv1c1 {fsPDdfv1c1}\n'
                              f'fsPDdfv1c2 {fsPDdfv1c2}')

                if (mu_max not in cfg.mu_maxes_to_plot
                    or Tref not in cfg.Trefs_to_plot
                    or Tfac not in cfg.Tfacs_to_plot
                    or virtualprog not in cfg.virtualprogs_to_plot
                    or SubLink_gal not in cfg.SubLink_gals_to_plot):
                        continue

                # create as-measured plots
                if cfg.plot_fKDE == 0:
                    fdat = np.zeros(3, dtype=dtype_fdat)
                    fdat[0] = ('RGct', fs_ct['fsRGct'][0])
                    fdat[1] = ('bin', fs_ct['fsbin'][0])
                    fdat[2] = ('mlt', fs_ct['fsmlt'][0])
                else:
                    fdat = np.zeros(7, dtype=dtype_fdat)
                    fdat[0] = ('RGct', fs_ct['fsRGct'][0])
                    fdat[1] = ('bin', fs_ct['fsbin'][0])
                    fdat[2] = ('mlt', fs_ct['fsmlt'][0])
                    fdat[3] = ('KDEc1', fsKDEc1)
                    fdat[4] = ('KDEc2', fsKDEc2)
                    fdat[5] = ('vldc1', fs_vldc1)
                    fdat[6] = ('vldc2', fs_vldc2)
                logging.debug(f'Plot a\nfdat\n{fdat}')
                createfvmplot(ilnum, snapnum, mu_min, mu_max, virtualprog,
                              SubLink_gal, Tref, Tfac, m_axis_max, fmin,
                              RGm_edges, fsRGfcn, m_ctrs, numgxy, fdat, 'a')

                # create PD plots
                fdat = np.zeros(6, dtype=dtype_fdat)
                fdat[0] = ('bin', fs_ct['fsbin'][0])
                fdat[1] = ('mlt', fs_ct['fsmlt'][0])
                fdat[2] = ('PDdfv0c1', fsPDdfv0c1)
                fdat[3] = ('PDdfv0c2', fsPDdfv0c2)
                fdat[4] = ('PDdfv1c1', fsPDdfv1c1)
                fdat[5] = ('PDdfv1c2', fsPDdfv1c2)
                logging.debug(f'Plot b\nfdat\n{fdat}')
                createfvmplot(ilnum, snapnum, mu_min, mu_max, virtualprog,
                              SubLink_gal, Tref, Tfac, m_axis_max, fmin,
                              RGm_edges, fsRGfcn, m_ctrs, numgxy, fdat, 'b')

                # create exact plots
                fdat = np.zeros(6, dtype=dtype_fdat)
                fdat[0] = ('bin', fs_ct['fsbin'][0])
                fdat[1] = ('mlt', fs_ct['fsmlt'][0])
                fdat[2] = ('1', fs_ct['fs1'][0])
                fdat[3] = ('2', fs_ct['fs2'][0])
                fdat[4] = ('3', fs_ct['fs3'][0])
                fdat[5] = ('4', fs_ct['fs4'][0])
                logging.debug(f'Plot c\nfdat\n{fdat}')
                createfvmplot(ilnum, snapnum, mu_min, mu_max, virtualprog,
                              SubLink_gal, Tref, Tfac, m_axis_max, fmin,
                              RGm_edges, fsRGfcn, m_ctrs, numgxy, fdat, 'c')
def createfvm_mlt_plot():
    """
    Create a plot comparing merger fractions across multiple z and simulations.
    """
    
    print(f'Creating multiple z/sim f vs m plot at ilnum {cfg.fvm_mlt_ilnums} '
          f'snapOG {cfg.fvm_mlt_snapsOG} snapTNG {cfg.fvm_mlt_snapsTNG} '
          f'mu_max {cfg.mu_maxes_to_plot} '
          f'virtualprog {cfg.virtualprogs_to_plot[0]} '
          f'SubLink_gal {cfg.SubLink_gals_to_plot[0]} '
          f'Tref {cfg.Trefs_to_plot[0]} Tfac {cfg.Tfacs_to_plot[0]} '
          f'mmin {cfg.mmin} mminvirt {cfg.mminvirt} '
          f'mbinsnumraw {cfg.mbinsnumraw} mmrglst3 {cfg.mmrglst3} '
          f'mlogspace {cfg.mlogspace} subhalostart {cfg.subhalostart} '
          f'subhalo_end {cfg.subhalo_end} ')
          
    logging.debug('Creating multiple z/sim f vs m plot at '
                  f'ilnum {cfg.fvm_mlt_ilnums} snapOG {cfg.fvm_mlt_snapsOG} '
                  f'snapTNG {cfg.fvm_mlt_snapsTNG} '
                  f'mu_max {cfg.mu_maxes_to_plot[0]} '
                  f'virtualprog {cfg.virtualprogs_to_plot[0]} '
                  f'SubLink_gal {cfg.SubLink_gals_to_plot[0]} '
                  f'Tref {cfg.Trefs_to_plot[0]} Tfac {cfg.Tfacs_to_plot[0]} '
                  f'mmin {cfg.mmin} mminvirt {cfg.mminvirt} '
                  f'mbinsnumraw {cfg.mbinsnumraw} mmrglst3 {cfg.mmrglst3} '
                  f'mlogspace {cfg.mlogspace} subhalostart {cfg.subhalostart} '
                  f'subhalo_end {cfg.subhalo_end} ')
    
    # confirm parameters correct
    if len(cfg.mu_maxes_to_plot) != 1:
        raise Exception("mu_max to plot must be exactly one value")
    if len(cfg.virtualprogs_to_plot) != 1:
        raise Exception("virtual prog to plot must be exactly one value")
    if len(cfg.SubLink_gals_to_plot) != 1:
        raise Exception("SubLink_gal to plot must be exactly one value")
    if len(cfg.Trefs_to_plot) != 1:
        raise Exception("Tref to plot must be exactly one value")
    if len(cfg.Tfacs_to_plot) != 1:
        raise Exception("Tfac to plot must be exactly one value")
    if len(cfg.fvm_mlt_snapsOG) != len(cfg.fvm_mlt_snapsTNG):
        raise Exception("Tfac to plot must be exactly one value")
    for i in range(len(cfg.fvm_mlt_snapsOG)):
        if (round(glb.zs[0][cfg.fvm_mlt_snapsOG[i]]*10)/10 != 
            round(glb.zs[1][cfg.fvm_mlt_snapsTNG[i]]*10)/10):
                raise Exception("TNG and OG z's must be equal")
                
    pltclrs = ['g', 'r', 'c', 'm', 'y', 'C0', 'C1', 'C4', 'C5', 'C6', 'C7']
    pltmrkrs = ['p', 's', 'o', 'X', 'D', 'v', '^', '<', '>']
    
    mu_min = 1 / cfg.mu_maxes_to_plot[0]
    
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))

    # create fs_mlt array
    if cfg.mmrglst3 == 0:
        mbinsnum = cfg.mbinsnumraw
    else:
        mbinsnum = cfg.mbinsnumraw - 2
    dtype_fs_mlt = [('ilnum', np.int16), ('snapnum', np.int16),
                    ('m_ctrs', np.float_, mbinsnum),
                    ('fs', np.float_, mbinsnum),
                    ('numgxy', np.float_, mbinsnum)]
    num_fs_mlt = 0
    for ilnum in cfg.fvm_mlt_ilnums:
        if ilnum == 1 or ilnum == 3:
            num_fs_mlt = num_fs_mlt + len(cfg.fvm_mlt_snapsOG)
        elif ilnum == 100 or ilnum == 300:
            num_fs_mlt = num_fs_mlt + len(cfg.fvm_mlt_snapsTNG)
    fs_mlt = np.zeros(num_fs_mlt, dtype=dtype_fs_mlt)
    logging.debug(f'fs_mlt dtype \n{fs_mlt.dtype.names}\nfs_mlt\n{fs_mlt}')

    # fill sim/z portion of fs_mlt array
    i_fs_mlt = 0
    for ilnum in cfg.fvm_mlt_ilnums:
        if ilnum == 1 or ilnum == 3:
            for snapnum in cfg.fvm_mlt_snapsOG:
                fs_mlt['ilnum'][i_fs_mlt] = ilnum
                fs_mlt['snapnum'][i_fs_mlt] = snapnum
                i_fs_mlt = i_fs_mlt + 1
        if ilnum == 100 or ilnum == 300:
            for snapnum in cfg.fvm_mlt_snapsTNG:
                fs_mlt['ilnum'][i_fs_mlt] = ilnum
                fs_mlt['snapnum'][i_fs_mlt] = snapnum
                i_fs_mlt = i_fs_mlt + 1

    # get plot data
    massmaxmax = 0
    fminmin = 1
    for i in range(len(fs_mlt)):
        ilnum = fs_mlt[i]['ilnum']
        if ilnum == 1 or ilnum == 3:
            snapOG = str(fs_mlt[i]['snapnum'])
            snapTNG = ''
            snapnum = snapOG
        elif ilnum == 100 or ilnum == 300:
            snapOG = ''
            snapTNG = str(fs_mlt[i]['snapnum'])
            snapnum = snapTNG
            
        # get maximum mass
        fcfg_mmax = (f'i{ilnum}sO{snapOG}sT{snapTNG}rs{rs}vs{vs}gs{gs}'
                     f'Trs{Trefstr}Tfs{Tfacs}mm{cfg.mmin:1.1f}'
                     f'mv{cfg.mminvirt:1.2f}'
                     f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
        with open(os.path.join('output', 'numerical', 'mmax',
                                'massmax' + fcfg_mmax + '.txt')) as f:
            massmax = float(f.read())
        if massmax > massmaxmax:
            massmaxmax = massmax
        logging.debug(f'massmax {massmax} massmaxmax {massmaxmax}')
        
        # get minimum f
        fcfg_fmin = (f'i{ilnum}sO{snapOG}sT{snapTNG}rs{rs}vs{vs}gs{gs}'
                     f'Trs{Trefstr}Tfs{Tfacs}mm{cfg.mmin:1.1f}'
                     f'mv{cfg.mminvirt:1.2f}mb{cfg.mbinsnumraw}'
                     f'mt{cfg.mmrglst3}ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                     f'do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}Km{cfg.KDEmult}'
                     f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
        with open(os.path.join('output', 'numerical', 'fmin',
                                'fmin' + fcfg_fmin + '.txt')) as f:
            fmin = float(f.read())
        if fmin < fminmin:
                fminmin = fmin
        logging.debug(f'fmin {fmin} fminmin {fminmin}')
        
        # set m_ctrs
        _, fs_mlt[i]['m_ctrs'], _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                         cfg.mbinsnumraw, cfg.mlogspace,
                                         cfg.mmrglst3)
    
        # set fs
        logging.debug('setting fs')
        fcfg_f = (f'i{ilnum}sO{snapOG}sT{snapTNG}rs{rs}vs{vs}gs{gs}'
                  f'Trs{Trefstr}Tfs{Tfacs}mm{cfg.mmin:1.1f}'
                  f'mv{cfg.mminvirt:1.2f}mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}'
                  f'ml{cfg.mlogspace}Rn{cfg.RGms_num}ss{cfg.subhalostart}'
                  f'se{cfg.subhalo_end}')
        with np.load(os.path.join('output', 'numerical', 'f',
                                  'fdat' + fcfg_f + '.npz')) as data:
            fs_ct_all = data['fs_ct_all']
                 
        fs_ct = fs_ct_all[(fs_ct_all['mu_min'] == mu_min)
                          & (fs_ct_all['mu_max'] == cfg.mu_maxes_to_plot[0])
                          & (fs_ct_all['virtualprog'] 
                             == cfg.virtualprogs_to_plot[0])
                          & (fs_ct_all['SubLink_gal'] 
                             == cfg.SubLink_gals_to_plot[0])
                          & (fs_ct_all['Tref'] == cfg.Trefs_to_plot[0])
                          & (fs_ct_all['Tfac'] == cfg.Tfacs_to_plot[0])][0]
        fs_mlt[i]['fs'] = fs_ct['fsbin']
        fs_mlt[i]['numgxy'] = fs_ct['numgxy']
    logging.debug(f'fs_mlt\n{fs_mlt}')
        
    # convert 0 and -1 data (non-error) values to NaN
    for i in range(num_fs_mlt):
        for j in range(mbinsnum):
            if fs_mlt[i]['fs'][j] <= 0:
                fs_mlt[i]['fs'][j] = np.nan
    logging.debug(f'fs_mlt, after 0s and -1s converted to NaN\n{fs_mlt}')
        
    # calculate error bars, if desired
    if cfg.ploterrorbars != 0:
        # create error array
        dtype_errs = [('num_mgrs', np.float_, mbinsnum), 
                      ('e_abs_lo', np.float_, mbinsnum),
                      ('e_abs_hi', np.float_, mbinsnum),
                      ('e_rel_lo', np.float_, mbinsnum),
                      ('e_rel_hi', np.float_, mbinsnum)]
        errs = np.zeros(num_fs_mlt, dtype=dtype_errs)
    
        # set errors
        for i in range(num_fs_mlt):
            for j in range(mbinsnum):
                if (fs_mlt[i]['numgxy'][j] > 0 and fs_mlt[i]['fs'][j] > 0
                    and fs_mlt[i]['fs'][j] <= 1):

                    num_mgrs = fs_mlt[i]['fs'][j] * fs_mlt[i]['numgxy'][j]                        
                    e_abs_lo, e_abs_hi = set_errbnds(num_mgrs, 
                                                     fs_mlt[i]['numgxy'][j])
                    e_rel_lo = fs_mlt[i]['fs'][j] - e_abs_lo
                    e_rel_hi = e_abs_hi - fs_mlt[i]['fs'][j]
                        
                    errs[i]['num_mgrs'][j] = num_mgrs
                    errs[i]['e_abs_lo'][j] = e_abs_lo
                    errs[i]['e_abs_hi'][j] = e_abs_hi
                    errs[i]['e_rel_lo'][j] = e_rel_lo      
                    errs[i]['e_rel_hi'][j] = e_rel_hi
        logging.debug(f'i {i} j {j}\nerrs {errs}')
                        
        # address nonpositive relative errors
        logging.debug(f'errs, before relative 0 removal\n{errs}')
        for i in range(num_fs_mlt):
            for j in range(mbinsnum):
                if errs[i]['e_rel_lo'][j] < 0:
                    warnings.warn(f"e_rel_lo < 0: {errs[i]['name']} mbin {j}, "
                                  'setting to np.nan')
                if errs[i]['e_rel_hi'][j] < 0:
                    warnings.warn(f"e_rel_hi < 0: {errs[i]['name']} mbin {j}, "
                                  'setting to np.nan')
                if errs[i]['e_rel_lo'][j] <= 0:
                    errs[i]['e_rel_lo'][j] = np.nan
                if errs[i]['e_rel_hi'][j] <= 0:
                    errs[i]['e_rel_hi'][j] = np.nan
        logging.debug(f'errs, after relative 0s converted to NaN\n{errs}')

    # plot fractions
    for i in range(len(fs_mlt)):
        if fs_mlt[i]['ilnum'] == 1 or fs_mlt[i]['ilnum'] == 3:
            il_lbl = f"Il-{fs_mlt[i]['ilnum']}"
            z_mlt = round(glb.zs[0][fs_mlt[i]['snapnum']]*10)/10
            line_alpha = 0.5
            # colorval = 
            markerfill = 'none'
            mrkrsize = 12
        elif fs_mlt[i]['ilnum'] == 100 or fs_mlt[i]['ilnum'] == 300:
            il_lbl = f"TNG{fs_mlt[i]['ilnum']}"
            z_mlt = round(glb.zs[1][fs_mlt[i]['snapnum']]*10)/10
            line_alpha = 1
            markerfill = pltclrs[i]
            mrkrsize = 6
        for j in range(len(cfg.fvm_mlt_snapsOG)):
            if z_mlt == round(glb.zs[0][cfg.fvm_mlt_snapsOG[j]]*10)/10:
                pltclr = pltclrs[j]
                break

        # plot points
        if cfg.ploterrorbars == 0:
            plt.plot(fs_mlt[i]['m_ctrs']*10**10, fs_mlt[i]['fs'],
                     alpha=line_alpha, color=pltclr,
                     label=f"{il_lbl}, z={z_mlt:.1f}", marker=pltmrkrs[i], 
                     markerfacecolor=markerfill, markersize=mrkrsize)
        else:
            if all(np.isnan(fs_mlt[i]['fs'])):
                yerrs = None
            else:
                yerrs = (errs[i]['e_rel_lo'], errs[i]['e_rel_hi'])
            plt.errorbar(fs_mlt[i]['m_ctrs']*10**10, fs_mlt[i]['fs'],
                         yerr=yerrs, alpha=line_alpha, color=pltclr,
                         label=f"{il_lbl}, z={z_mlt:.1f}", marker=pltmrkrs[i],
                         markerfacecolor=markerfill, markersize=mrkrsize)

    # set plot m axis max
    logging.debug('m axis max and R-G edges')
    if cfg.m_axis_maxmanual == 1:
        with open(os.path.join('output', 'numerical',
                                'm_axis_maxmanual.txt')) as f:
            m_axis_max = float(f.read())
    else:
        m_axis_max = massmaxmax + cfg.bin_pdng
    logging.debug(f'm_axis_max {m_axis_max}')
    
    # plot legend/titles/etc
    plt.legend(bbox_to_anchor=(0.99, 0.01), loc='lower right',
               borderaxespad=0.0, prop={'size':12}, handletextpad=0.2,
               labelspacing=0)
    plt.xlabel(r'$M_{\ast}$ [$M_{sun}$]')
    plt.ylabel('Merger fraction')
    if cfg.mlogspace:
        plt.xscale('log')
    plt.yscale('log')
    plt.xlim(cfg.mmin*10**10, m_axis_max*10**10)
    plt.ylim(0.01, 1)
    mpl.rcParams['errorbar.capsize'] = 4
    
    plt.tight_layout(pad=0.2)
    
    if cfg.plot_tofile:
        if cfg.functions[cfg.fnum] == 'createpubplots':
            pathname = os.path.join('output', 'graphical', 'pub')
        else:
            pathname = os.path.join('output', 'graphical', 'f_vs_m')
        if (not os.path.exists(pathname)):
            os.makedirs(pathname)
        plt.savefig(os.path.join(pathname,
                    f'fmltrl{mu_min:1.2f}ru{cfg.mu_maxes_to_plot[0]:02d}'
                    f'vp{cfg.virtualprogs_to_plot[0]}'
                    f'sg{cfg.SubLink_gals_to_plot[0]}'
                    f'Tr{cfg.Trefs_to_plot[0][0]}'
                    f'Tf{cfg.Tfacs_to_plot[0]:0.1f}'
                    f'mb{cfg.mbinsnumraw}mlt{cfg.mmrglst3}'
                    f'ss{cfg.subhalostart}se{cfg.subhalo_end}.png'))
    if cfg.plot_toconsole:
        plt.show()
    plt.clf()
    plt.close()

def createfvmratioplot(ilnums, zs, mu_maxes, virtualprogs, SubLink_gals,
                       Trefs, Tfacs, pds, dfvs, values, ratios, values_avg, 
                       ratios_avg):
    """
    Create a plot showing ratio of binary to multiple mergers, using either
    measured values, or those based on probability densities. 

    Parameters
    ----------
    ilnums : int
        Illustris run numbers.
    zs : float
        Redshifts.
    mu_maxes : int
        Minimum merger mass ratios.
    virtualprogs : boolean
        True if virtual progenitors created, false if not.
    SubLink_gals : boolean
        True if SubLink_gal trees used, False if SubLInk trees used.
    Trefs : string
        The reference used to determine T.
    Tfacs : float
        The factor by which T is multiplied.
    pds : boolean
        True if using probability density ratios, false if not.
    dfvs : boolean
        Null if not using probability density ratios. If using PDs, True if
        using values divided by the valid fraction, False if not.
    values : ndarray
        Binary merger values.
    ratios : ndarray
        Ratio of multiple to binary values.
    values_avg : ndarray
        Binary merger values used to compute average line.
    ratios : ndarray
        Ratio of multiple to binary values used to compute average line.
    """

    print(f'Creating f vs m ratio plot at ilnums {ilnums} zs {zs} '
          f'mu_maxes {mu_maxes} virtualprogs {virtualprogs} '
          f'SubLink_gals {SubLink_gals} Trefs {Trefs} Tfacs {Tfacs} '
          f'pds {pds} dfvs {dfvs}')
    
    logging.debug(f'createfvmratioplot: ilnums {ilnums} zs {zs} '
                  f'mu_maxes {mu_maxes} virtualprogs {virtualprogs} '
                  f'SubLin  k_gals {SubLink_gals} Trefs {Trefs} Tfacs {Tfacs} '
                  f'pds {pds} dfvs {dfvs} values\n{values}\nratios\n{ratios}')

    # convert 0 and -1 data (non-error) values to NaN
    for i in range(len(values)):
        if values[i] == -1 and ratios[i] != -1:
            raise ValueError ('value == -1, but ratio != -1')
        elif ratios[i] == -1 and values[i] != -1:
            raise ValueError ('ratio == -1, but value != -1')
        elif values[i] == -1 and ratios[i] == -1:
            ratios[i], values[i] = np.nan, np.nan
    logging.debug(f"After -1's converted to NaN: values {values}\n"
                  f'ratios {ratios}')
    
    # plot data
    pltbin_edges, _, _ = setbinlims(0, 1, 50, 0, 0)
    plt.hist2d(values, ratios, norm=mpl.colors.LogNorm(), bins=pltbin_edges)

    # plot average ratio, if option set
    if cfg.plot_ratio_avgs == True:
        logging.debug('Adding average ratio')
    
        # fill ratio bins
        logging.debug('fb_bin edges, centers')
        fb_bin_edges, fb_bin_ctrs, _ = setbinlims(0, 1, cfg.ratio_avg_numbins,
                                                  0, 0)
        fb_bin_idcs = np.digitize(values_avg, bins=fb_bin_edges)
        binned_ratios = [[] for  x in range(cfg.ratio_avg_numbins)]
        for i in range(len(values_avg)):
            if fb_bin_idcs[i] <= cfg.ratio_avg_numbins:
                binned_ratios[fb_bin_idcs[i]-1].append(ratios_avg[i])
        logging.debug(f'fb_bin_idcs {fb_bin_idcs}\nbinned_ratios {binned_ratios}')
        
        # compute bin averages
        bin_avgs = np.full(cfg.ratio_avg_numbins, np.nan)
        for i in range(len(binned_ratios)):
            if len(binned_ratios[i]) > 0:
                bin_avgs[i] = sum(binned_ratios[i])/len(binned_ratios[i])
        logging.debug(f'bin_avgs {bin_avgs}')
    
        plt.plot(fb_bin_ctrs, bin_avgs, color='r', label='Average')
        plt.legend(loc='lower right')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar()
    plt.xlabel('$f_t$')
    plt.ylabel('$f_m/f_t$')

    # create titles and figure text
    if cfg.functions[cfg.fnum] != 'createpubplots':
        zstr = '['
        for i in range(len(zs)):
            zstr = zstr + str(zs[i]) + ', '
            if len(zs) >= 6 and i == len(zs) / 2:
                zstr = zstr + '\n     '
        zstr = zstr[:-2] +']'

        plt.title('Ratio of Multiple to Binary Fractions vs Binary Fraction')
        figtxt = (f'Illustris: {ilnums}\n$z$: {zstr}\n$\mu_{{max}}$: '
                  f'{mu_maxes}\nVirtual Progs: {virtualprogs}\n'
                  f'SubLink_gal: {SubLink_gals}\n'
                  f'$T_{{ref}}$: {Trefs}\n$T_{{fac}}$: {Tfacs}\n'
                  f'$\lambda=c$: {pds}\n'
                  f'$/f_v$: {dfvs}')
        plt.figtext(0.88, 0.4, figtxt)
    
    if cfg.plot_tofile == 1:
        ilfn = ''.join(map(str, ilnums))
        zfn = ''.join(map(str, zs))
        mu_maxfn = ''.join(map(str, mu_maxes))
        vsfn = ''.join(map(str, virtualprogs))
        gsfn = ''.join(map(str, SubLink_gals))
        Trfn = ''
        for Tref in Trefs:
            Trfn = Trfn + Tref[0]
        Tffn = ''.join(map(str, Tfacs))
        pds = ''.join(map(str, pds))
        dfvs = ''.join(map(str, dfvs))
        
        if cfg.functions[cfg.fnum] == 'createpubplots':
            pathname = os.path.join('output', 'graphical', 'pub')
        else:
            pathname = os.path.join('output', 'graphical', 'ratio')
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        plt.savefig(
                os.path.join(
                        pathname, 'fri' + ilfn + 'z' + zfn + 'mu' + mu_maxfn
                        + 'v' + vsfn + 's' + gsfn + 'Tr' + Trfn + 'Tf' + Tffn 
                        + f'p{pds}d{dfvs}mm{cfg.mmin:1.1f}'
                        f'mv{cfg.mminvirt:1.2f}mb{cfg.mbinsnumraw}'
                        f'mt{cfg.mmrglst3}ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                        f'do{cfg.dtbinwdthopt:1.1f}ss{cfg.subhalostart}'
                        f'se{cfg.subhalo_end}.png'),
                bbox_inches='tight')
    if cfg.plot_toconsole:
        plt.show()
    plt.clf()
    plt.close()
    
def createfvmratioplots():
    """
    Create multiple plots showing the ratio of binary to multiple mergers,
    using either measured values, or those based on probability densities. 
    """
    
    print('Running createfvmratioplots')
    logging.debug(f'ilnums {cfg.ilnums_ratio} '
                  f'snapnumsOGratio {cfg.snapnumsOGratio} '
                  f'snapnumsTNGratio {cfg.snapnumsTNGratio} '
                  f'mu_maxes {cfg.mu_maxes} virtualprogs {cfg.virtualprogs} '
                  f'SubLink_gals {cfg.SubLink_gals} Trs {cfg.Trefs} '
                  f'Tfs {cfg.Tfacs} mmin {cfg.mmin} mminvirt {cfg.mminvirt} '
                  f'mbinsnumraw {cfg.mbinsnumraw} mmrglst3 {cfg.mmrglst3} '
                  f'mlogspace {cfg.mlogspace} dtbinwdthopt {cfg.dtbinwdthopt} '
                  f'subhalostart {cfg.subhalostart} '
                  f'subhalo_end {cfg.subhalo_end} '
                  f'mu_maxes_to_plot {cfg.mu_maxes_to_plot_ratio} '
                  f'virtualprogs_to_plot {cfg.virtualprogs_to_plot_ratio} '
                  f'SubLink_gals_to_plot {cfg.SubLink_gals_to_plot_ratio} '
                  f'Trefs_to_plot {cfg.Trefs_to_plot_ratio} '
                  f'Tfacs_to_plot {cfg.Tfacs_to_plot_ratio}')
    
    # get data file config strings
    sOGs = ''.join(map(str, cfg.snapnumsOGratio))
    sTNGs = ''.join(map(str, cfg.snapnumsTNGratio))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))

    # get num snapnums
    numsnapnums = 0
    for ilnum in cfg.ilnums_ratio:
        if ilnum == 1 or ilnum == 3:
            numsnapnums += len(cfg.snapnumsOGratio)
        elif ilnum == 100 or ilnum == 300:
            numsnapnums += len(cfg.snapnumsTNGratio)
    logging.debug(f'num snapnums {numsnapnums}')

    if cfg.mmrglst3 == 1:
        numbins = cfg.mbinsnumraw - 2
    else:
        numbins = cfg.mbinsnumraw    

    # create arrays
    lenRs = (numsnapnums * len(cfg.mu_maxes_to_plot_ratio)
             * len(cfg.virtualprogs_to_plot_ratio)
             * len(cfg.SubLink_gals_to_plot_ratio)
             * len(cfg.Trefs_to_plot_ratio) * len(cfg.Tfacs_to_plot_ratio) * 3)
    dtypeRs = [('ilnum', np.int16), ('snapnum', np.int16),
               ('mu_max', np.float16), ('virtualprog', np.bool_),
               ('SubLink_gal', np.bool_), ('Tref', np.unicode_, 10),
               ('Tfac', np.float16), ('pd', np.bool_), ('dfv', np.bool_),
               ('values', np.float_, numbins), ('ratios', np.float_, numbins)]
    ratios = np.full(lenRs, -1, dtype=dtypeRs)
    rspd0 = np.full(numbins, -1, dtype=np.float_)
    rsPD1dfv0 = np.full(numbins, -1, dtype=np.float_)
    rsPD1dfv1 = np.full(numbins, -1, dtype=np.float_)
    iR = 0

    # fill ratio array
    for ilnum in cfg.ilnums_ratio:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOGratio
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNGratio
            
        for snapnum in snapnums:
            if ilnum == 1 or ilnum == 3:
                sOGs = snapnum
                sTNGs = ''
            elif ilnum == 100 or ilnum == 300:
                sOGs = ''
                sTNGs = snapnum
            _, _, ilrun, _, _, _ = get_run_info(ilnum)
            
            logging.debug(f'ilnum {ilnum} snapnum {snapnum}')
            print(f'Getting data at ilnum {ilnum} snapnum {snapnum}')
            
            # get non-dt derived f data
            fcfg_f = (f'i{ilnum}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}'
                      f'Trs{Trefstr}Tfs{Tfacs}mm{cfg.mmin:1.1f}'
                      f'mv{cfg.mminvirt:1.2f}mb{cfg.mbinsnumraw}'
                      f'mt{cfg.mmrglst3}ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                      f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'f',
                                      'fdat' + fcfg_f + '.npz')) as data:
                fs_ct_all = data['fs_ct_all']
            logging.debug(f'fs_ct_all\n{fs_ct_all}')
            
            for mu_max, virtualprog, SubLink_gal, Tref, Tfac in it.product(
                    cfg.mu_maxes_to_plot_ratio, cfg.virtualprogs_to_plot_ratio,
                    cfg.SubLink_gals_to_plot_ratio, cfg.Trefs_to_plot_ratio,
                    cfg.Tfacs_to_plot_ratio):

                # required due to galaxy with > 40 mergers found with these
                # parameters creating setdts probability table too large to be
                # calculated
                if (ilnum == 1 and snapnum == 75 and mu_max == 10 and
                        SubLink_gal == 1 and Tref == 'merger' and Tfac == 2):
                    continue

                logging.debug(f'mu_max {mu_max} virtualprog '
                              f'{virtualprog} SubLink_gal {SubLink_gal} '
                              f'Tref {Tref} Tfac {Tfac}')

                # get non-dt-derived f data
                fs_ct = fs_ct_all[(fs_ct_all['mu_max'] == mu_max)
                                  & (fs_ct_all['virtualprog'] == virtualprog)
                                  & (fs_ct_all['SubLink_gal'] == SubLink_gal)
                                  & (fs_ct_all['Tref'] == Tref)
                                  & (fs_ct_all['Tfac'] == Tfac)]
                fsPD0bin = fs_ct['fsbin'][0]
                fsPD0mlt = fs_ct['fsmlt'][0]
                logging.debug(f'fs_ct\n{fs_ct}\nfsPD0bin {fsPD0bin}\n'
                              f'fsPD0mlt {fsPD0mlt}')

                # get dt-derived f data
                fcfg_dt = (f'i{ilnum}s{snapnum}r{mu_max}v{virtualprog}'
                           f'g{SubLink_gal}Tr{Tref[0]}Tf{Tfac}'
                           f'mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                           f'mb{cfg.mbinsnumraw}mt{cfg.mmrglst3}'
                           f'ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                           f'do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}'
                           f'Km{cfg.KDEmult}ss{cfg.subhalostart}'
                           f'se{cfg.subhalo_end}')
                with np.load(os.path.join('output', 'numerical', 'dt', ilrun,
                                          'dt_dat' + fcfg_dt + '.npz'))\
                        as data:
                    fsKDEc1 = data['fsKDEc1']
                    fsKDEc2 = data['fsKDEc2']
                    fsPD1dfv0c1 = data['fsPDdfv0c1']
                    fsPD1dfv0c2 = data['fsPDdfv0c2']
                    fsPD1dfv1c1 = data['fsPDdfv1c1']
                    fsPD1dfv1c2 = data['fsPDdfv1c2']
                logging.debug(f'fsKDEc1 {fsKDEc1}\nfsKDEc2 {fsKDEc2}\n'
                              f'fsPD1dfv0c1 {fsPD1dfv0c1}\nfsPD1dfv0c2 '
                              f'{fsPD1dfv0c2}\nfsPD1dfv1c1 {fsPD1dfv1c1}\n'
                              f'fsPD1dfv1c2 {fsPD1dfv1c2}')
                              
                # set ratios
                for i in range(numbins):
                    # bin, mult
                    if fsPD0bin[i] == 0:
                        if fsPD0mlt[i] == 0:
                            rspd0[i] = 0
                        else:
                            raise ValueError ('Mult != 0 when Bin = 0')
                    else:
                        rspd0[i] = fsPD0mlt[i] / fsPD0bin[i]
                        
                    # PDF, dfv0
                    if fsPD1dfv0c1[i] == -1:
                        if fsPD1dfv0c2[i] == -1:
                            rsPD1dfv0[i] = -1
                        else:
                            raise ValueError ('PDF dfv0 c2 != -1 when 1c = -1')
                    elif fsPD1dfv0c1[i] == 0:
                        if fsPD1dfv0c2[i] == 0:
                            rsPD1dfv0[i] = 0
                        else:
                            raise ValueError ('PDF dfv0 c2 != 0 when 1c == 0')
                    else:
                        rsPD1dfv0[i] = fsPD1dfv0c2[i] / fsPD1dfv0c1[i]
                        
                    # PDF, dfv1
                    if fsPD1dfv1c1[i] == -1:
                        if fsPD1dfv1c2[i] == -1:
                            rsPD1dfv1[i] = -1
                        else:
                            raise ValueError ('PDF dfv1 c2 != -1 when 1c = -1')
                    elif fsPD1dfv1c1[i] == 0:
                        if fsPD1dfv1c2[i] == 0:
                            rsPD1dfv1[i] = 0
                        else:
                            raise ValueError ('PDF dfv1 c2 != 0 when 1c == 0')
                    else:
                        rsPD1dfv1[i] = fsPD1dfv1c2[i] / fsPD1dfv1c1[i]
                logging.debug(f'rspd0 {rspd0}\nrsPD1dfv0 {rsPD1dfv0}\n'
                              f'rsPD1dfv1 {rsPD1dfv1}')                                                
                        
                # insert values           
                ratios[iR] = ((ilnum, snapnum, mu_max, virtualprog, 
                               SubLink_gal, Tref, Tfac, 0, 0, fsPD0bin, rspd0))
                iR += 1
                ratios[iR] = ((ilnum, snapnum, mu_max, virtualprog, 
                               SubLink_gal, Tref, Tfac, 1, 0, fsPD1dfv0c1,
                               rsPD1dfv0))
                iR += 1
                ratios[iR] = ((ilnum, snapnum, mu_max, virtualprog, 
                               SubLink_gal, Tref, Tfac, 1, 1, fsPD1dfv1c1,
                               rsPD1dfv1))
                iR += 1                    
                logging.debug(f'iR {iR}\nratios\n{ratios}')
    
    # create plots
    zsOG = []
    zsTNG = []
    if 1 in cfg.ilnums_ratio or 3 in cfg.ilnums_ratio:
        for i in range(len(cfg.snapnumsOGratio)):
            z = round(glb.zs[0][cfg.snapnumsOGratio[i]], 1)
            if z - math.floor(z) == 0:
                z = int(z)
            zsOG.append(z)
    if 100 in cfg.ilnums_ratio or 300 in cfg.ilnums_ratio:
        for i in range(len(cfg.snapnumsTNGratio)):
            z = round(glb.zs[1][cfg.snapnumsTNGratio[i]], 1)
            if z - math.floor(z) == 0:
                z = int(z)
            zsTNG.append(z)
    zsOG.sort(reverse=True)
    zsTNG.sort(reverse=True)
    logging.debug(f'zsOG {zsOG} zsTNG {zsTNG}')
    
    # create plots for each combination of ilnum, z, vp, S_g, Tref, and Tfac
    if cfg.functions[cfg.fnum] != 'createpubplots':
        for ilnum, virtualprog, SubLink_gal, Tref, Tfac, in it.product(
                cfg.ilnums_ratio, cfg.virtualprogs_to_plot_ratio,
                cfg.SubLink_gals_to_plot_ratio, cfg.Trefs_to_plot_ratio,
                cfg.Tfacs_to_plot_ratio):
    
            logging.debug(f'Creating plots at ilnum {ilnum} virtualprog '
                          f'{virtualprog} SubLink_gal {SubLink_gal} Tref '
                          f'{Tref} Tfac {Tfac}')
        
            # required due to galaxy with > 40 mergers found with these 
            # parameters creating setdts p table too large to be calculated
            if (ilnum == 1 and SubLink_gal == 1 and Tref == 'merger'
                    and Tfac == 2):
                continue
                 
            rs_plt = ratios[(ratios['ilnum'] == ilnum)
                            & (ratios['virtualprog'] == virtualprog)
                            & (ratios['SubLink_gal'] == SubLink_gal)
                            & (ratios['Tref'] == Tref)
                            & (ratios['Tfac'] == Tfac)]
            logging.debug(f'rs_plt\n{rs_plt}')
    
            if ilnum == 1 or ilnum == 3:
                zs_ratio = zsOG
            elif ilnum == 100 or ilnum == 300:
                zs_ratio = zsTNG
    
            # get final values and ratios
            rsPD0dfv0r0 = (rs_plt[(rs_plt['pd'] == 0) & (rs_plt['dfv'] == 0)]
                            ['values'].flatten())
            rsPD0dfv0r1 = (rs_plt[(rs_plt['pd'] == 0) & (rs_plt['dfv'] == 0)]
                            ['ratios'].flatten())
            rsPD1dfv0r0 = (rs_plt[(rs_plt['pd'] == 1) & (rs_plt['dfv'] == 0)]
                            ['values'].flatten())
            rsPD1dfv0r1 = (rs_plt[(rs_plt['pd'] == 1) & (rs_plt['dfv'] == 0)]
                            ['ratios'].flatten())
            rsPD1dfv1r0 = (rs_plt[(rs_plt['pd'] == 1) & (rs_plt['dfv'] == 1)]
                            ['values'].flatten())
            rsPD1dfv1r1 = (rs_plt[(rs_plt['pd'] == 1) & (rs_plt['dfv'] == 1)]
                            ['ratios'].flatten())
            logging.debug(f'rsPD0dfv0r0 {rsPD0dfv0r0}\nrsPD0dfv0r1 '
                          f'{rsPD0dfv0r1}\nrsPD1dfv0r0 {rsPD1dfv0r0}\n'
                          f'rsPD1dfv0r1 {rsPD1dfv0r1}\nrsPD1dfv1r0 '
                          f'{rsPD1dfv1r0}\nrsPD1dfv1r1 {rsPD1dfv1r1}')
            
            # create plots
            createfvmratioplot([ilnum], zs_ratio, cfg.mu_maxes_to_plot_ratio,
                                [virtualprog], [SubLink_gal], [Tref], [Tfac],
                                [0], [], rsPD0dfv0r0, rsPD0dfv0r1,
                                rsPD0dfv0r0, rsPD0dfv0r1)
            createfvmratioplot([ilnum], zs_ratio, cfg.mu_maxes_to_plot_ratio,
                                [virtualprog], [SubLink_gal], [Tref], [Tfac],
                                [1], [0], rsPD1dfv0r0, rsPD1dfv0r1,
                                rsPD0dfv0r0, rsPD0dfv0r1)
            createfvmratioplot([ilnum], zs_ratio, cfg.mu_maxes_to_plot_ratio,
                                [virtualprog], [SubLink_gal], [Tref], [Tfac],
                                [1], [1], rsPD1dfv1r0, rsPD1dfv1r1,
                                rsPD0dfv0r0, rsPD0dfv0r1)
    
    # create ratio plot from results using all parameters
    # set zs
    if zsOG != zsTNG:
        raise Exception("OG and TNG redshifts must be equal")
    zs_ratio = zsOG
    
    # flatten values and ratios, create plots
    rsPD0r0 = (ratios[(ratios['pd'] == 0)]['values'].flatten())
    rsPD0r1 = (ratios[(ratios['pd'] == 0)]['ratios'].flatten())
    rsPD1r0 = (ratios[(ratios['pd'] == 1)]['values'].flatten())
    rsPD1r1 = (ratios[(ratios['pd'] == 1)]['ratios'].flatten())
    
    # create plots
    createfvmratioplot(cfg.ilnums_ratio, zs_ratio, cfg.mu_maxes_to_plot_ratio, 
                        cfg.virtualprogs_to_plot_ratio,
                        cfg.SubLink_gals_to_plot_ratio,
                        cfg.Trefs_to_plot_ratio, cfg.Tfacs_to_plot_ratio,
                        [0], [], rsPD0r0, rsPD0r1, rsPD0r0, rsPD0r1)
    createfvmratioplot(cfg.ilnums_ratio, zs_ratio, cfg.mu_maxes_to_plot_ratio, 
                        cfg.virtualprogs_to_plot_ratio,
                        cfg.SubLink_gals_to_plot_ratio, 
                        cfg.Trefs_to_plot_ratio, cfg.Tfacs_to_plot_ratio,
                        [1], [0, 1], rsPD1r0, rsPD1r1, rsPD0r0, rsPD0r1)
    
def create_dt1d_plot(
        ilnum, snapnum, mu_min, mu_max, virtualprog, SubLink_gal, Tref, Tfac,
        cl, cml, nrm, m_edge_lo, m_edge_hi, fvld1c, fvld2c, fKDE, R, dt_ctrs,
        dt_edges, dt_wdths, dts, dts1p, dts2p, dts1n, dts2n, f_xs, df_dtc_dfv0,
        df_dtc_dfv1, df_Rc_dfv0, df_Rc_dfv1, pdfKDE):
    """
    Create dt-related 1D plot.

    Parameters
    ----------
    ilnum : int
        Illustris run number.
    snapnum : int
        Starting analysis snapnum.
    mu_min : float
        Minimum acceptable valid merger mass ratio.
    mu_max : float
        Maximum acceptable valid merger mass ratio.
    virtualprog : boolean
        True if virtual progenitors created, false if not
    SubLink_gal : boolean
        True if SubLink_gal trees used, false if SubLInk trees used
    Tref : string
        The reference used to determine T.
    Tfac : float
        The factor by which T is multiplied.
    cl : int
        The cl-th closest merger being analyzed.
    cml : bool
        True if cumulative plot, False if not.
    nrm : bool
        True if normalized plot, False if not.
    m_edge_lo : float
        Value of the lower mass bin edge.
    m_edge_hi : float
        Value of the upper mass bin edge.
    fvld1c : float
        Fraction of subhalos with valid first-closest dts.
    fvld2c : float
        Fraction of subhalos with valid second-closest dts.
    fKDE : float
        Merger fraction, as determined by Kernel Density Estimate.
    R : float
        Merger rate.
    dt_ctrs : ndarray of floats
        Center of each dt bin.
    dt_edges : ndarray of floats
        Edges of the dt bins.
    dt_wdths : ndarray of floats
        Width of each dt bin.
    dts : ndarray of delta-ts
        Delta-t values to be plotted.
    dts1p : ndarray of floats
        Delta-ts to 1st previous merger (None if cumulative or normalized).
        non-normalized).
    dts2p : ndarray of floats
        Delta-ts to 2nd previous merger (None if cumulative or normalized).
    dts1n : ndarray of floats
        Delta-ts to 1st next merger (None if cumulative or normalized).
    dts2n : ndarray of floats
        Delta-ts to 2nd next merger (None if cumulative or normalized).
    f_xs : ndarray of floats
        PDF/CDF x values, None if non-normalized.
    df_dtc_dfv0 : ndarray of floats
        PDF/CDF, constant time between events, None if non-normalized.
    df_dtc_dfv1 : ndarray of floats
        PDF/CDF, constant time between events, / frac valid, None if
        non-normalized.
    df_Rc_dfv0 : ndarray of floats
        PDF/CDF, constant event rate, None if non-normalized.
    df_Rc_dfv1 : ndarray of floats
        PDF/CDF, constant event rate, / frac valid, None if
        non-normalized.
    pdfKDE : ndarray of floats
        Kernel Density Estimate, None if non-normalized or cumulative.
    """

    dictnum, _, ilrun, _, _, _ = get_run_info(ilnum)

    print('test')

    # get T
    T = -1
    if Tref == 'analysis' or Tref == 'merger':
        T = glb.Tsnys[dictnum][snapnum] * Tfac
    elif Tref == 'snapwidth':
        T = glb.Tsnps[dictnum][snapnum] * Tfac
    logging.debug(f'T {T}')

    # plot data
    plt.bar(dt_ctrs, dts, width=dt_wdths, color='0.5', edgecolor='0')
    if cml == 0 and nrm == 0:
        plt.plot(dt_ctrs, dts1p, 'r' , label='First previous')
        plt.plot(dt_ctrs, dts1n, 'm' , label='First next')
        if cfg.functions[cfg.fnum] == 'createpubplots' and cl == 2:
            plt.plot(dt_ctrs, dts2p, 'b' , label='Second previous')
            plt.plot(dt_ctrs, dts2n, 'c' , label='Second next')
    if nrm == 1:
        plt.plot(f_xs, df_dtc_dfv0, 'b', label='$\Delta t_m=1/R$')
        plt.plot(f_xs, df_dtc_dfv1, 'g', label='$\Delta t_m=F/R$')
        plt.plot(f_xs, df_Rc_dfv0, 'r', label='$\lambda=2R$')
        plt.plot(f_xs, df_Rc_dfv1, 'orange', label='$\lambda=2R/F$')
        if cml == 0:
            plt.plot(f_xs, pdfKDE, 'purple', label='KDE')

    ylabel = '# Galaxies'
    if nrm == 1:
        ylabel = 'Galaxy density [$Gyr^{-1}$]'

    if cfg.functions[cfg.fnum] != 'createpubplots':
        # plot text
        st_st = '# Galaxies'
        st_cl = '$1^{st}$'
        st_cml, st_nrm = '', ''
        if cl == 2:
            st_cl = '$2^{nd}$'
        if cml == 1:
            st_cml = ', Cumulative'
        if nrm == 1:
            st_st = r'$\rho_{Gal}$'
            st_nrm = ', Normalized'
    
        plt.suptitle(f'{st_st} vs Time to {st_cl}-'
                     f'closest Merger{st_cml}{st_nrm}', fontsize=14)
        plt.title(f'{ilrun}, $z$ = {glb.zs[dictnum][snapnum]:.1f}, '
                  f'$\mu$ = {mu_min}-{mu_max}')
    
        KDEtxt = ''
        if cml == 0 and nrm == 1:
            KDEtxt = f'$f_{{KDE}}$: {fKDE:0.3f}\n'
            
        if platform.node().startswith('jupyter'):
            figtxt_y = 0.10
            if cml == 0:
                if nrm == 0:
                    figtxt_x = 0.83
                else:
                    figtxt_x = 0.79
            else:
                if nrm == 0:
                    figtxt_x = 0.98
                else:
                    figtxt_x = 0.79
        else:
            figtxt_y = 0.12
            if cml == 0:
                if nrm == 0:
                    figtxt_x = 0.81
                else:
                    figtxt_x = 0.78
            else:
                if nrm == 0:
                    figtxt_x = 0.98
                else:
                    figtxt_x = 0.78
        
        fvtxt = f'$f_{{v, 1c}}$: {fvld1c:0.2f}'
        if cl == 2:
            fvtxt += f'; $f_{{v, 2c}}$: {fvld2c:0.2f}'
        
        if R == -1:
            Rval = 'N/A'
        else:
            Rval = f'{R:0.3f}'    
        
        figtxt = (f'Virtual Progs: {str(bool(virtualprog))[0]}\n'
                  f'SubLink_gal: {str(bool(SubLink_gal))[0]}\n'
                  f'$T_{{ref}}$: {Tref}\n$T_{{fac}}$: {Tfac}\n{fvtxt}\n'
                  f'$R$: {Rval}\n{KDEtxt}'
                  r'$M_{\ast}$ ($M_{{sun}}$):'
                  f'\n{m_edge_lo*10**10:0.2E} - {m_edge_hi*10**10:0.2E}')
        plt.figtext(figtxt_x, figtxt_y, figtxt, linespacing=1)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                   borderaxespad=0.0)
    else:
        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right',
                   borderaxespad=0.0)
    if cl == 1:
        plt.xlabel('$\Delta t_1$, the time to the closest merger [Gyr]')
    elif cl == 2:
        plt.xlabel('$\Delta t_2$, the time to the second-closest merger [Gyr]')
    plt.ylabel(ylabel)
    plt.xlim(0, dt_edges[-1])
    plt.plot([T/2, T/2], [0, plt.axis()[3]], color='k', linestyle='--')
    plt.tight_layout(pad=0.2)

    if cfg.plot_tofile == 1:
        if cfg.functions[cfg.fnum] == 'createpubplots':
            pathname = os.path.join('output', 'graphical', 'pub')
        else:
            pathname = os.path.join('output', 'graphical', 'dt-related', '1D',
                                    ilrun, f'sn{snapnum:03d}'
                                    f'rl{mu_min:1.2f}ru{mu_max:02d}')
        if (not os.path.exists(pathname)):
            os.makedirs(pathname)
        plt.savefig(
                os.path.join(pathname, f'dtv{virtualprog:01d}'
                             f's{SubLink_gal:01d}Tr{Tref[0]}Tf{Tfac:0.1f}'
                             f'cl{cl}c{cml}n{nrm}ml{m_edge_lo:2.2f}'
                             f'mh{m_edge_hi:2.2f}ss{cfg.subhalostart}'
                             f'se{cfg.subhalo_end}.png'),
                bbox_inches='tight')
    if cfg.plot_toconsole:
        plt.show()
    plt.clf()
    plt.close()

def create_dt2d_plot(ilnum, snapnum, mu_min, mu_max, virtualprog,
                     SubLink_gal, Tref, Tfac, m_edge_lo, m_edge_hi, dt_edges,
                     dts_c1, dts_c2):
    """
    Create dt-related 2D histogram.

    Parameters
    ----------
    ilnum : int
        Illustris run number.
    snapnum : int
        Starting analysis snapnum.
    mu_min : float
        Minimum acceptable valid merger mass ratio.
    mu_max : float
        Maximum acceptable valid merger mass ratio.
    virtualprog : boolean
        True if virtual progenitors created, false if not
    SubLink_gal : boolean
        True if SubLink_gal trees used, false if SubLInk trees used
    Tref : string
        The reference used to determine T.
    Tfac : float
        The factor by which T is multiplied.
    m_edge_lo : float
        Value of the lower mass bin edge.
    m_edge_hi : float
        Value of the upper mass bin edge.
    dt_edges : ndarray of floats
        Edges of the dt bins.
    dts_c1 : ndarray
        1st-closest delta-t values to be plotted.
    dts_c2 : ndarray
        2st-closest delta-t values to be plotted.
    """

    dictnum, _, ilrun, _, _, _ = get_run_info(ilnum)

    logging.debug(f'create_dt2d_plot: ilnum {ilnum} snapnum {snapnum} '
                  f'mu_max {mu_max} virtualprog {virtualprog} '
                  f'SubLink_gal {SubLink_gal} Tref {Tref} Tfac {Tfac} '
                  f'dt_edges\n{dt_edges}\ndts_c1\n{dts_c1}\ndts_c2\n{dts_c2}')
    
    # plot data
    plt.hist2d(dts_c1, dts_c2, norm=mpl.colors.LogNorm(), bins=dt_edges)
    plt.colorbar()

    # plot text
    if cfg.functions[cfg.fnum] != 'createpubplots':
        plt.suptitle('# Galaxies, $2^{nd}$- vs $1^{st}$-closest Merger',
                     fontsize=14)
        plt.title(f'{ilrun}, $z$ = {glb.zs[dictnum][snapnum]:.1f}, '
                  f'$\mu$ = {mu_min}-{mu_max}')
        figtxt = (f'Virtual Progs: {str(bool(virtualprog))[0]}\n'
                  f'SubLink_gal: {str(bool(SubLink_gal))[0]}\n'
                  f'$T_{{ref}}$: {Tref}\n$T_{{fac}}$: {Tfac}\n'
                  r'$M_{\ast}$ $M_{sun})$:'
                  f'\n{m_edge_lo*10**10:0.2E} -\n{m_edge_hi*10**10:0.2E}')
        plt.figtext(0.96, 0.12, figtxt)
        
    plt.xlabel('$\Delta t_1$, the time to the closest merger [Gyr]')
    plt.ylabel('$\Delta t_2$, time to 2nd-closest merger [Gyr]')
    plt.tight_layout(pad=0.2)

    if cfg.plot_tofile == 1:
        if cfg.functions[cfg.fnum] == 'createpubplots':
            pathname = os.path.join('output', 'graphical', 'pub')
        else:
            pathname = os.path.join('output', 'graphical', 'dt-related', '2D',
                                    ilrun, f'sn{snapnum:03d}'
                                    f'rl{mu_min:1.2f}ru{mu_max:02d}')
        if (not os.path.exists(pathname)):
            os.makedirs(pathname)
        plt.savefig(
            os.path.join(pathname, f'dtv{virtualprog:01d}s{SubLink_gal:01d}'
                         f'Tr{Tref[0]}Tf{Tfac:0.1f}ml{m_edge_lo:2.2f}'
                         f'mh{m_edge_hi:2.2f}ss{cfg.subhalostart}'
                         f'se{cfg.subhalo_end}.png'), bbox_inches='tight')
    if cfg.plot_toconsole:
        plt.show()
    plt.clf()
    plt.close()

def create_dtplots():
    """
    Create multiple dt plots.
    """

    print('Running create_dtplots')

    # get config strings
    ils = ''.join(map(str, cfg.ilnums))
    sOGs = ''.join(map(str, cfg.snapnumsOG))
    sTNGs = ''.join(map(str, cfg.snapnumsTNG))
    rs = ''.join(map(str, cfg.mu_maxes))
    vs = ''.join(map(str, cfg.virtualprogs))
    gs = ''.join(map(str, cfg.SubLink_gals))
    Trefstr = ''
    for Tref in cfg.Trefs:
        Trefstr = Trefstr + Tref[0]
    Tfacs = ''.join(map(str, cfg.Tfacs))
    
    # set mass max and bin info
    logging.debug('m bins')
    fcfg_mmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                 f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                 f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'mmax',
                           'massmax' + fcfg_mmax + '.txt')) as f:
        massmax = float(f.read())
    logging.debug(f'massmax {massmax}')
    m_edges, m_ctrs, _ = setbinlims(cfg.mmin, massmax + cfg.bin_pdng,
                                    cfg.mbinsnumraw, cfg.mlogspace,
                                    cfg.mmrglst3)

    # get maximum measured dt value
    fcfg_dtmax = (f'i{ils}sO{sOGs}sT{sTNGs}rs{rs}vs{vs}gs{gs}Trs{Trefstr}'
                  f'Tfs{Tfacs}mm{cfg.mmin:1.1f}mv{cfg.mminvirt:1.2f}'
                  f'ss{cfg.subhalostart}se{cfg.subhalo_end}')
    with open(os.path.join('output', 'numerical', 'dtmax',
                           'dtmax' + fcfg_dtmax + '.txt')) as f:
        dtmax = float(f.read())
    logging.debug(f'dtmax {dtmax}')

    # create plots
    for ilnum in cfg.ilnums:
        if ilnum == 1 or ilnum == 3:
            snapnums = cfg.snapnumsOG
        elif ilnum == 100 or ilnum == 300:
            snapnums = cfg.snapnumsTNG
        dictnum, basepath, ilrun, snapnummax, _, numsubgroups \
            = get_run_info(ilnum)

        for snapnum, mu_max, virtualprog, SubLink_gal, Tref, Tfac in \
                it.product(snapnums, cfg.mu_maxes_to_plot,
                           cfg.virtualprogs_to_plot, cfg.SubLink_gals_to_plot, 
                           cfg.Trefs_to_plot, cfg.Tfacs_to_plot_dt):
            logging.debug(f'ilnum {ilnum} snapnum {snapnum} mu_max {mu_max} '
                          f'virtualprog {virtualprog} SubLink_gal '
                          f'{SubLink_gal} Tref {Tref} Tfac {Tfac}')

            # set T
            if Tref == 'analysis' or Tref == 'merger':
                T = glb.Tsnys[dictnum][snapnum] * Tfac
            elif Tref == 'snapwidth':
                T = glb.Tsnps[dictnum][snapnum] * Tfac
            logging.debug(f'T {T}')

            # get dt data
            fcfg_dt = (f'i{ilnum}s{snapnum}r{mu_max}v{virtualprog}'
                       f'g{SubLink_gal}Tr{Tref[0]}Tf{Tfac}mm{cfg.mmin:1.1f}'
                       f'mv{cfg.mminvirt:1.2f}mb{cfg.mbinsnumraw}'
                       f'mt{cfg.mmrglst3}ml{cfg.mlogspace}Rn{cfg.RGms_num}'
                       f'do{cfg.dtbinwdthopt:1.1f}fn{cfg.fxs_num}'
                       f'Km{cfg.KDEmult}ss{cfg.subhalostart}'
                       f'se{cfg.subhalo_end}')
            with np.load(os.path.join('output', 'numerical', 'dt', ilrun,
                                      'dt_dat' + fcfg_dt + '.npz')) as data:
                Rs_dfv0 = data['Rs_dfv0']
                fs_vldc1, fs_vldc2 = data['fs_vldc1'], data['fs_vldc2']
                fsKDEc1, fsKDEc2 = data['fsKDEc1'], data['fsKDEc2']
                dts_c0n0 = data['dts_c0n0']
                dts_c1yc2n, dts_c1nc2n = data['dts_c1yc2n'], data['dts_c1nc2n']
                dts_c1_c0n1 = data['dts_c1_c0n1']
                dts_c1_c1n0 = data['dts_c1_c1n0']
                dts_c1_c1n1 = data['dts_c1_c1n1']
                dts_c2_c0n1 = data['dts_c2_c0n1']
                dts_c2_c1n0 = data['dts_c2_c1n0']
                dts_c2_c1n1 = data['dts_c2_c1n1']
                dts1p, dts2p = data['dts1p'], data['dts2p']
                dts1n, dts2n = data['dts1n'], data['dts2n']
                pdfs_dtc_dfv0c1 = data['pdfs_dtc_dfv0c1']
                pdfs_dtc_dfv0c2 = data['pdfs_dtc_dfv0c2']
                pdfs_dtc_dfv1c1 = data['pdfs_dtc_dfv1c1']
                pdfs_dtc_dfv1c2 = data['pdfs_dtc_dfv1c2']
                pdfs_Rc_dfv0c1 = data['pdfs_Rc_dfv0c1']
                pdfs_Rc_dfv0c2 = data['pdfs_Rc_dfv0c2']
                pdfs_Rc_dfv1c1 = data['pdfs_Rc_dfv1c1']
                pdfs_Rc_dfv1c2 = data['pdfs_Rc_dfv1c2']
                pdfsKDEc1 = data['pdfsKDEc1']
                pdfsKDEc2 = data['pdfsKDEc2']
                cdfs_dtc_dfv0c1 = data['cdfs_dtc_dfv0c1']
                cdfs_dtc_dfv0c2 = data['cdfs_dtc_dfv0c2']
                cdfs_dtc_dfv1c1 = data['cdfs_dtc_dfv1c1']
                cdfs_dtc_dfv1c2 = data['cdfs_dtc_dfv1c2']
                cdfs_Rc_dfv0c1 = data['cdfs_Rc_dfv0c1']
                cdfs_Rc_dfv0c2 = data['cdfs_Rc_dfv0c2']
                cdfs_Rc_dfv1c1 = data['cdfs_Rc_dfv1c1']
                cdfs_Rc_dfv1c2 = data['cdfs_Rc_dfv1c2']
            logging.debug(
                    f'Rs_dfv0 {Rs_dfv0}\ndts_c0n0 {dts_c0n0}\n'
                    f'dts_c1yc2n {dts_c1yc2n}\ndts_c1nc2n {dts_c1nc2n}\n'
                    f'dts_c1_c0n1 {dts_c1_c0n1}\ndts_c1_c1n0 {dts_c1_c1n0}\n'
                    f'dts_c1_c1n1 {dts_c1_c1n1}\ndts_c2_c0n1 {dts_c2_c0n1}\n'
                    f'dts_c2_c1n0 {dts_c2_c1n0}\ndts_c2_c1n1 {dts_c2_c1n1}\n'
                    f'dts1p {dts1p}\ndts2p {dts2p}\n'
                    f'dts1n {dts1n}\ndts2n {dts2n}\n'
                    f'pdfs_dtc_dfv0c1 {pdfs_dtc_dfv0c1}\n'
                    f'pdfs_dtc_dfv0c2 {pdfs_dtc_dfv0c2}\n'
                    f'pdfs_dtc_dfv1c1 {pdfs_dtc_dfv1c1}\n'
                    f'pdfs_dtc_dfv1c2 {pdfs_dtc_dfv1c2}\n'
                    f'pdfs_Rc_dfv0c1 {pdfs_Rc_dfv0c1}\n'
                    f'pdfs_Rc_dfv0c2 {pdfs_Rc_dfv0c2}\n'
                    f'pdfs_Rc_dfv1c1 {pdfs_Rc_dfv1c1}\n'
                    f'pdfs_Rc_dfv1c2 {pdfs_Rc_dfv1c2}\n'
                    f'pdfsKDEc1 {pdfsKDEc1}\npdfsKDEc2 {pdfsKDEc2}\n'
                    f'cdfs_dtc_dfv0c1 {cdfs_dtc_dfv0c1}\n'
                    f'cdfs_dtc_dfv0c2 {cdfs_dtc_dfv0c2}\n'
                    f'cdfs_dtc_dfv1c1 {cdfs_dtc_dfv1c1}\n'
                    f'cdfs_dtc_dfv1c2 {cdfs_dtc_dfv1c2}\n'
                    f'cdfs_Rc_dfv0c1 {cdfs_Rc_dfv0c1}\n'
                    f'cdfs_Rc_dfv0c2 {cdfs_Rc_dfv0c2}\n'
                    f'cdfs_Rc_dfv1c1 {cdfs_Rc_dfv1c1}\n'
                    f'cdfs_Rc_dfv1c2 {cdfs_Rc_dfv1c2}')

            # set dt bin info
            dtbinwidth = T/2 / max(1, round(T/2 / cfg.dtbinwdthopt))
            dtbinsnum = math.ceil((dtmax + cfg.bin_pdng) / dtbinwidth)
            dt_edgemax = dtbinwidth * dtbinsnum
            logging.debug(f'dtbinwidth {dtbinwidth} dtbinsnum {dtbinsnum} '
                          f'dt_edgemax {dt_edgemax}')
            dt_edges, dt_ctrs, dt_wdths = setbinlims(0, dt_edgemax,
                                                     dtbinsnum, 0, 0)
            
            # set function independent values
            logging.debug('f_xs')
            _, f_xs, _ = setbinlims(0, dt_edgemax, cfg.fxs_num, 0, 0)

            # set mass bins to plot
            if -1 in cfg.dt_mbins_to_plot:
                mbins_to_plot = range(len(m_ctrs))
            else:
                mbins_to_plot = cfg.dt_mbins_to_plot

            # create 1D plots
            if cfg.plotcml == 0: cml_rng = [0]
            else: cml_rng = [0, 1]
            for cl, cml, nrm, mbin_num in it.product([1, 2], cml_rng, [0, 1],
                                                     mbins_to_plot):
                print(f'Creating 1D plot at ilnum {ilnum} snapnum {snapnum} '
                      f'mu_max {mu_max} virtualprog {virtualprog} '
                      f'SubLink_gal {SubLink_gal} Tref {Tref} Tfac {Tfac} '
                      f'cl {cl} cml {cml} nrm {nrm} mbin_num {mbin_num}')

                R = Rs_dfv0[mbin_num]
                fvldc1, fvldc2 = fs_vldc1[mbin_num], fs_vldc2[mbin_num]
                fKDE, pdfKDE = None, None
                dt1p, dt2p, dt1n, dt2n = None, None, None, None
                df_dtc_dfv0, df_dtc_dfv1 = None, None
                df_Rc_dfv0, df_Rc_dfv1 = None, None
                if cl == 1:
                    if cml == 0:
                        if nrm == 0:
                            dts = (np.sum(dts_c0n0[mbin_num], axis=0)
                                   + dts_c1yc2n[mbin_num])
                            dt1p = dts1p[mbin_num]
                            dt2p = dts2p[mbin_num]
                            dt1n = dts1n[mbin_num]
                            dt2n = dts2n[mbin_num]
                        elif nrm == 1:
                            fKDE = fsKDEc1[mbin_num]
                            dts = dts_c1_c0n1[mbin_num]
                            df_dtc_dfv0 = pdfs_dtc_dfv0c1[mbin_num]
                            df_dtc_dfv1 = pdfs_dtc_dfv1c1[mbin_num]
                            df_Rc_dfv0 = pdfs_Rc_dfv0c1[mbin_num]
                            df_Rc_dfv1 = pdfs_Rc_dfv1c1[mbin_num]
                            pdfKDE = pdfsKDEc1[mbin_num]
                    elif cml == 1:
                        if nrm == 0:
                            dts = dts_c1_c1n0[mbin_num]
                        elif nrm == 1:
                            dts = dts_c1_c1n1[mbin_num]
                            df_dtc_dfv0 = cdfs_dtc_dfv0c1[mbin_num]
                            df_dtc_dfv1 = cdfs_dtc_dfv1c1[mbin_num]
                            df_Rc_dfv0 = cdfs_Rc_dfv0c1[mbin_num]
                            df_Rc_dfv1 = cdfs_Rc_dfv1c1[mbin_num]
                elif cl == 2:
                    if cml == 0:
                        if nrm == 0:
                            dts = np.sum(dts_c0n0[mbin_num], axis=1)
                            dt1p = dts1p[mbin_num]
                            dt2p = dts2p[mbin_num]
                            dt1n = dts1n[mbin_num]
                            dt2n = dts2n[mbin_num]
                        elif nrm == 1:
                            fKDE = fsKDEc2[mbin_num]
                            dts = dts_c2_c0n1[mbin_num]
                            df_dtc_dfv0 = pdfs_dtc_dfv0c2[mbin_num]
                            df_dtc_dfv1 = pdfs_dtc_dfv1c2[mbin_num]
                            df_Rc_dfv0 = pdfs_Rc_dfv0c2[mbin_num]
                            df_Rc_dfv1 = pdfs_Rc_dfv1c2[mbin_num]
                            pdfKDE = pdfsKDEc2[mbin_num]
                    elif cml == 1:
                        if nrm == 0:
                            dts = dts_c2_c1n0[mbin_num]
                        elif nrm == 1:
                            dts = dts_c2_c1n1[mbin_num]
                            df_dtc_dfv0 = cdfs_dtc_dfv0c2[mbin_num]
                            df_dtc_dfv1 = cdfs_dtc_dfv1c2[mbin_num]
                            df_Rc_dfv0 = cdfs_Rc_dfv0c2[mbin_num]
                            df_Rc_dfv1 = cdfs_Rc_dfv1c2[mbin_num]
                logging.debug(
                        f'ilnum {ilnum} snapnum {snapnum} mu_max '
                        f'{mu_max} virtualprog {virtualprog} SubLink_gal '
                        f'{SubLink_gal} Tref {Tref} Tfac {Tfac} cl {cl} cml '
                        f'{cml} nrm {nrm} mbin_num {mbin_num} T {T} m_lo '
                        f'{m_edges[mbin_num]} m_hi {m_edges[mbin_num+1]} '
                        f'fvldc1 {fvldc1} fvldc2 {fvldc2} fKDE {fKDE} '
                        f'R {R}\ndt_ctrs\n{dt_ctrs}\ndt_edges\n{dt_edges}\n'
                        f'dt_wdths\n{dt_wdths}\ndts\n{dts}\ndt1p\n{dt1p}\n'
                        f'dt2p\n{dt2p}\ndt1n\n{dt1n}\ndt2n\n{dt2n}\n'
                        f'f_xs\n{f_xs}\ndf_dtc_dfv0\n{df_dtc_dfv0}\n'
                        f'df_dtc_dfv1\n{df_dtc_dfv1}\n'
                        f'df_Rc_dfv0\n{df_Rc_dfv0}\ndf_Rc_dfv1\n{df_Rc_dfv1}\n'
                        f'pdfKDE\n{pdfKDE}')
                create_dt1d_plot(
                        ilnum, snapnum, 1/mu_max, mu_max, virtualprog,
                        SubLink_gal, Tref, Tfac, cl, cml, nrm,
                        m_edges[mbin_num], m_edges[mbin_num+1], fvldc1, fvldc2,
                        fKDE, R, dt_ctrs, dt_edges, dt_wdths, dts, dt1p, dt2p,
                        dt1n, dt2n, f_xs, df_dtc_dfv0, df_dtc_dfv1, df_Rc_dfv0,
                        df_Rc_dfv1, pdfKDE)

            # create 2D plots
            for mbin_num in mbins_to_plot:
                print(f'Creating 2D plot at ilnum {ilnum} snapnum {snapnum} '
                      f'mu_max {mu_max} virtualprog {virtualprog} '
                      f'SubLink_gal {SubLink_gal} Tref {Tref} Tfac {Tfac} '
                      f'mbin_num {mbin_num}')

                # get 2D array length
                arylen_dt2d = 0
                for dtbin_c1 in range(len(dt_ctrs)):
                    for dtbin_c2 in range(len(dt_ctrs)):
                        arylen_dt2d += int(round(
                                dts_c0n0[mbin_num][dtbin_c2][dtbin_c1]))
                logging.debug(f'arylen_dt2d {arylen_dt2d}')

                # fill 2D arrays
                dts_2dc1 = np.zeros(arylen_dt2d)
                dts_2dc2 = np.zeros(arylen_dt2d)
                i_dt2d = 0
                for dtbin_c1 in range(len(dt_ctrs)):
                    for dtbin_c2 in range(len(dt_ctrs)):
                        num_dts = int(round(
                                dts_c0n0[mbin_num][dtbin_c2][dtbin_c1]))
                        for i in range(num_dts):
                            dts_2dc1[i_dt2d] = dt_ctrs[dtbin_c1]
                            dts_2dc2[i_dt2d] = dt_ctrs[dtbin_c2]
                            i_dt2d += 1
                logging.debug(
                        f'ilnum {ilnum} snapnum {snapnum} mu_max '
                        f'{mu_max} virtualprog {virtualprog} SubLink_gal '
                        f'{SubLink_gal} Tref {Tref} Tfac {Tfac} mbin_num '
                        f'{mbin_num} dt_edges\n{dt_edges}\ndt_ctrs\n{dt_ctrs}'
                        f'\ndts_2dc1\n{dts_2dc1}\ndts_2dc2\n{dts_2dc2}')
                create_dt2d_plot(
                        ilnum, snapnum, 1/mu_max, mu_max, virtualprog,
                        SubLink_gal, Tref, Tfac, m_edges[mbin_num],
                        m_edges[mbin_num+1], dt_edges, dts_2dc1, dts_2dc2)

def createpubplots():
    """
    Create plots formatted for publication.
    """
    
    print('Running create_pub_plots')
    
    plt.rcParams.update({'font.size': 12})
    createfvmplots()
    createfvm_mlt_plot()
    createfvmratioplots()
    create_dtplots()
    
def test():
    """
    Test code.
    """

    print('Running test')
        
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
