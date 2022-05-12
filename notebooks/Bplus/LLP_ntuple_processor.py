import awkward as ak
import numpy as np
import hist
from coffea import processor
from coffea.nanoevents.methods import vector, candidate


class LLP_ntuple_processor(processor.ProcessorABC):
    """
    This class is used to process the ntuples created by the LLP ntuple producer.
    """

    def process(self, events):
        """
        This function is used to process the ntuples into histograms.
        :param events:
        :return:
        """

        # create a dictionary for storing dimensions of different parts of the detector
        detector_dimensions = {
            'csc': {},
            'dt': {},
            'cms': {},
        }

        # fill detector_dimensions with the dimensions of the different parts of the detector
        s = 1e2  # scale factor
        detector_dimensions['csc']['zmin'] = s * 5.5
        detector_dimensions['csc']['zmax'] = s * 10.
        detector_dimensions['csc']['rmin'] = s * 0.
        detector_dimensions['csc']['rmax'] = s * 7.
        detector_dimensions['dt']['zmin'] = s * 0.
        detector_dimensions['dt']['zmax'] = s * 6.5
        detector_dimensions['dt']['rmin'] = s * 4.
        detector_dimensions['dt']['rmax'] = s * 7.5

        detector_dimensions['cms']['zmin'] = s * 0.
        detector_dimensions['cms']['zmax'] = s * 12.
        detector_dimensions['cms']['rmin'] = s * 0.
        detector_dimensions['cms']['rmax'] = s * 8.

        dataset = events.metadata['dataset']
        sumw = ak.sum(events.genWeight)

        out = {
            dataset: {
                "entries": len(events),
                "sumw": sumw,
            }
        }

        def R(x, y):
            """
            computes radius
            :param x:
            :param y:
            :return:
            """
            return np.sqrt(x ** 2 + y ** 2)

        # simple if statement that assigns a variable 'pid' to 9900015 or 1000023 if any of the events has one of
        # those particles
        if ak.any(events.gParticleId == 9900015): pid = 9900015
        if ak.any(events.gParticleId == 1000023): pid = 1000023

        # we need the decay vertices of the llps but events doesn't have that information. but we do have access to
        # the id of the mother particles (gParticleMotherId) as well as production vertices of all particles (
        # gParticleProdVertexX, gParticleProdVertexY, gParticleProdVertexZ) Make a mask of the particles whose mother
        # is the llp
        mother_llp_mask = events.gParticleMotherId == pid
        # However, each llp will have multiple products, so events.gParticleProdVertexX[mother_llp_mask] will have
        # duplicates To remove the duplicates we can use a trick that uses ak.argmax to find the index of the first
        # occurrence of each truth value Make an awkward array of the indices of the first occurrence of each truth
        # value within the second level of the mask
        llp_daughter_index = ak.argmax(mother_llp_mask, axis=1, keepdims=True)
        # Now we can use llp_daughter_index to get the indexes of the llps themselves using gParticleMotherIndex
        llp_index = events.gParticleMotherIndex[llp_daughter_index]

        # make a zip of thw llp decay vertices and PtEtaPhiMLorentzVector
        gen_llps = ak.zip(
            {
                'x': events.gParticleProdVertexX[llp_daughter_index],
                'y': events.gParticleProdVertexY[llp_daughter_index],
                'z': events.gParticleProdVertexZ[llp_daughter_index],
                'r': R(events.gParticleProdVertexX[llp_daughter_index],
                       events.gParticleProdVertexY[llp_daughter_index]),
                'pt': events.gParticlePt[llp_index],
                'phi': events.gParticlePhi[llp_index],
                'eta': events.gParticleEta[llp_index],
                'mass': ak.ones_like(events.gParticlePt[llp_index]),
                'E': events.gParticleE[llp_index],
            },
            with_name='PtEtaPhiMLorentzVector',
            behavior=vector.behavior,
        )

        # do the same with cscRechitCluster
        csc_cls = ak.zip(
            {
                'x': events.cscRechitClusterX,
                'y': events.cscRechitClusterY,
                'z': events.cscRechitClusterZ,
                'r': R(events.cscRechitClusterX, events.cscRechitClusterY),
                'pt': ak.ones_like(events.cscRechitClusterPhi),
                'phi': events.cscRechitClusterPhi,
                'eta': events.cscRechitClusterEta,
                'mass': ak.ones_like(events.cscRechitClusterPhi),
                'E': ak.ones_like(events.cscRechitClusterPhi),
            },
            with_name='PtEtaPhiMLorentzVector',
            behavior=vector.behavior,
        )

        # do exactly the same with dtRechitCluster
        dt_cls = ak.zip(
            {
                'x': events.dtRechitClusterX,
                'y': events.dtRechitClusterY,
                'z': events.dtRechitClusterZ,
                'r': R(events.dtRechitClusterX, events.dtRechitClusterY),
                'pt': ak.ones_like(events.dtRechitClusterPhi),
                'phi': events.dtRechitClusterPhi,
                'eta': events.dtRechitClusterEta,
                'mass': ak.ones_like(events.dtRechitClusterPhi),
                'E': ak.ones_like(events.dtRechitClusterPhi),
            },
            with_name='PtEtaPhiMLorentzVector',
            behavior=vector.behavior,
        )

        # do exactly the same with cscRechits
        csc_rechits = ak.zip(
            {
                'x': events.cscRechitsX,
                'y': events.cscRechitsY,
                'z': events.cscRechitsZ,
                'r': R(events.cscRechitsX, events.cscRechitsY),
                'pt': ak.ones_like(events.cscRechitsPhi),
                'phi': events.cscRechitsPhi,
                'eta': events.cscRechitsEta,
                'mass': ak.ones_like(events.cscRechitsPhi),
                'E': ak.ones_like(events.cscRechitsPhi),
            },
            with_name='PtEtaPhiMLorentzVector',
            behavior=vector.behavior,
        )

        # do exactly the same with dtRechits
        dt_rechits = ak.zip(
            {
                'x': events.dtRechitX,
                'y': events.dtRechitY,
                'z': events.dtRechitZ,
                'r': R(events.dtRechitX, events.dtRechitY),
                'pt': ak.ones_like(events.dtRechitPhi),
                'phi': events.dtRechitPhi,
                'eta': events.dtRechitEta,
                'mass': ak.ones_like(events.dtRechitPhi),
                'E': ak.ones_like(events.dtRechitPhi),
            },
            with_name='PtEtaPhiMLorentzVector',
            behavior=vector.behavior,
        )

        def csc_cut(v):
            """
            takes a PtEtaPhiMLorentzVector and returns a PtEtaPhiMLorentzVector with cuts in the csc region
            :param v:
            :return:
            """
            cut = (
                    (abs(v.eta) < 2.4) &
                    (abs(v.z) > detector_dimensions['csc']['zmin']) & (abs(v.z) < detector_dimensions['csc']['zmax']) &
                    (v.r < detector_dimensions['csc']['rmax'])
            )
            return v[cut]

        def dt_cut(v):
            """
            takes a PtEtaPhiMLorentzVector and returns a PtEtaPhiMLorentzVector with cuts in the dt region
            :param v:
            :return:
            """
            cut = (
                    (v.r > detector_dimensions['dt']['rmin']) & (v.r < detector_dimensions['dt']['rmax']) &
                    (abs(v.z) < detector_dimensions['dt']['zmax'])
            )
            return v[cut]

        def cartesian_delta_r(v1, v2):
            """
            takes two PtEtaPhiMLorentzVector awkward arrays and returns all the combinations of
            pairs and deltaR's
            :param v1:
            :param v2:
            :return:
            """
            # ensure that the key and the name of the variable are the same for consistency
            # this is probably bad practice, but it works
            v1name = [k for k, v in locals().items() if v is v1][0]
            v2name = [k for k, v in locals().items() if v is v2][0]
            pairs = ak.cartesian({v1name: v1, v2name: v2}, axis=1, nested=True)
            delta_r = pairs.v1.delta_r(pairs.v2)
            return ak.zip(
                {
                    'delta_r': delta_r,
                    v1name: pairs.v1,
                    v2name: pairs.v2,
                },
            )

        def delta_r_cut(v, threshold):
            """
            takes pairs and returns pairs with delta_r < threshold
            :param v:
            :param threshold:
            :return:
            """
            cut = v.delta_r < threshold
            return v[cut]

        def init_2d_hist(bins):
            """
            takes a number of bins and returns a 2D histogram with those bins within the cms region
            :param bins:
            :return:
            """
            dims = detector_dimensions['cms']
            return (
                hist.Hist.new
                    .Reg(bins, dims['zmin'], dims['zmax'], name='z', label='z [cm]')
                    .Reg(bins, dims['rmin'], dims['rmax'], name='r', label='r [cm]')
                    .Double()
            )

        def fill_2d_hist(hist, v):
            """
            takes a 2D histogram and a PtEtaPhiMLorentzVector and fills it
            :param hist:
            :param v:
            :return:
            """
            z = abs(ak.flatten(v.z, axis=None))
            r = abs(ak.flatten(v.r, axis=None))
            hist.fill(z, r)

        def init_fill_2d_hist(bins, v):
            """
            takes a number of bins and a PtEtaPhiMLorentzVector, then initializes and fills a 2D histogram with those bins
            :param bins:
            :param v:
            :return:
            """
            hist = init_2d_hist(bins)
            fill_2d_hist(hist, v)
            return hist

        def init_1d_hist(bins, domain, name, label):
            """
            takes a number of bins, a domain, a name, and a label and returns a 1D histogram with those bins
            :param bins:
            :param domain:
            :param name:
            :param label:
            :return:
            """
            return hist.Hist.new.Reg(bins, domain[0], domain[1], name=name, label=label).Double()

        def fill_1d_hist(hist, arr):
            """
            takes a 1D histogram and a flat array and fills it
            :param hist:
            :param arr:
            :return:
            """
            hist.fill(ak.flatten(arr, axis=None))

        def init_fill_1d_hist(bins, domain, arr, name, label):
            """
            takes a number of bins, a domain, a flat array, a name, and a label and initializes and fills a 1D
            histogram with those bins
            :param bins:
            :param domain:
            :param arr:
            :param name:
            :param label:
            :return:
            """
            hist = init_1d_hist(bins, domain, name, label)
            fill_1d_hist(hist, arr)
            return hist

        # cut gen_llps to detector regions
        csc_llps = csc_cut(gen_llps)
        dt_llps = dt_cut(gen_llps)

        # delta r pairs between cls and gen_llps
        csc_cls_gen_llp_pairs = cartesian_delta_r(csc_cls, gen_llps)
        dt_cls_gen_llp_pairs = cartesian_delta_r(dt_cls, gen_llps)

        # delta r's between cls and llps in region
        csc_cls_csc_llps_pairs = cartesian_delta_r(csc_cls, csc_llps)
        dt_cls_dt_llps_pairs = cartesian_delta_r(dt_cls, dt_llps)

        # delta r's between rechits and gen_llps
        csc_rechits_gen_llp_pairs = cartesian_delta_r(csc_rechits, gen_llps)
        dt_rechits_gen_llp_pairs = cartesian_delta_r(dt_rechits, gen_llps)

        # delta r's between rechits and llps in region
        csc_rechits_csc_llps_pairs = cartesian_delta_r(csc_rechits, csc_llps)
        dt_rechits_dt_llps_pairs = cartesian_delta_r(dt_rechits, dt_llps)

        # same as before but with delta_r < threshold
        threshold = 0.4
        csc_cls_gen_llp_pairs_dr_cut = delta_r_cut(csc_cls_gen_llp_pairs, threshold)
        dt_cls_gen_llp_pairs_dr_cut = delta_r_cut(dt_cls_gen_llp_pairs, threshold)
        csc_cls_csc_llps_pairs_dr_cut = delta_r_cut(csc_cls_csc_llps_pairs, threshold)
        dt_cls_dt_llps_pairs_dr_cut = delta_r_cut(dt_cls_dt_llps_pairs, threshold)
        csc_rechits_gen_llp_pairs_dr_cut = delta_r_cut(csc_rechits_gen_llp_pairs, threshold)
        dt_rechits_gen_llp_pairs_dr_cut = delta_r_cut(dt_rechits_gen_llp_pairs, threshold)
        csc_rechits_csc_llps_pairs_dr_cut = delta_r_cut(csc_rechits_csc_llps_pairs, threshold)
        dt_rechits_dt_llps_pairs_dr_cut = delta_r_cut(dt_rechits_dt_llps_pairs, threshold)

        bins = 30
        hists1d = {}
        # fill energy histograms
        hists1d['E_llp_cms'] = init_fill_1d_hist(bins, (0, 200), gen_llps.E, 'E_llp_cms', 'E_llp [GeV]')
        hists1d['E_llp_csc'] = init_fill_1d_hist(bins, (0, 10000), csc_llps.E, 'E_llp_csc', 'E_llp [GeV]')
        hists1d['E_llp_dt'] = init_fill_1d_hist(bins, (0, 10000), dt_llps.E, 'E_llp_dt', 'E_llp [GeV]')

        # fill pt histograms
        hists1d['pt_llp_cms'] = init_fill_1d_hist(bins, (0, 200), gen_llps.pt, 'pt_llp_cms', 'pt_llp [GeV]')
        hists1d['pt_llp_csc'] = init_fill_1d_hist(bins, (0, 200), csc_llps.pt, 'pt_llp_csc', 'pt_llp [GeV]')
        hists1d['pt_llp_dt'] = init_fill_1d_hist(bins, (0, 200), dt_llps.pt, 'pt_llp_dt', 'pt_llp [GeV]')

        # fill eta histograms
        hists1d['eta_llp_cms'] = init_fill_1d_hist(bins, (-5, 5), gen_llps.eta, 'eta_llp_cms', 'eta_llp')
        hists1d['eta_llp_csc'] = init_fill_1d_hist(bins, (-5, 5), csc_llps.eta, 'eta_llp_csc', 'eta_llp')
        hists1d['eta_llp_dt'] = init_fill_1d_hist(bins, (-5, 5), dt_llps.eta, 'eta_llp_dt', 'eta_llp')

        # fill phi histograms
        hists1d['phi_llp_cms'] = init_fill_1d_hist(bins, (-3.14, 3.14), gen_llps.phi, 'phi_llp_cms', 'phi_llp')
        hists1d['phi_llp_csc'] = init_fill_1d_hist(bins, (-3.14, 3.14), csc_llps.phi, 'phi_llp_csc', 'phi_llp')
        hists1d['phi_llp_dt'] = init_fill_1d_hist(bins, (-3.14, 3.14), dt_llps.phi, 'phi_llp_dt', 'phi_llp')

        out[dataset]['hists1d'] = hists1d
        return out

    def postprocess(self, accumulator):
        return accumulator
