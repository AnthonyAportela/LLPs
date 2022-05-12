from coffea.nanoevents import BaseSchema
import mplhep as hep
import matplotlib.pyplot as plt
from LLP_ntuple_processor import LLP_ntuple_processor
from coffea import processor

# digging up
def rootAdds(directory):
    my_file = open(directory, "r")
    data = my_file.read().strip()
    data_into_list = data.split("\n")
    my_file.close()
    return data_into_list

fileset = {}
fileset['hnl'] = rootAdds('rootAdds/BToHNL_MuonAndHNLGenFilter_mHNL1p0_ctau1000.txt')
fileset['phi'] = rootAdds('rootAdds/BToKPhi_MuonGenFilter_mPhi1p0_ctau1000.txt')

out = processor.run_uproot_job(
    fileset,
    treename="ntuples/llp",
    processor_instance=LLP_ntuple_processor(),
    executor=processor.futures_executor,
    executor_args={"schema": BaseSchema, "workers": 6},
    maxchunks=30,
)

for hist in out['phi']['hists1d']:
    if 'cms' in hist:
        hep.histplot(out['phi']['hists1d'][hist])
        plt.title(f'{hist}')
        plt.show()
