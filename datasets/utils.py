import os
import megfile

def load_pairs_from_cache_file(fname, root=''):
    assert megfile.smart_isfile(fname), "cannot parse pairs from {:s}, file does not exist".format(fname)
    with megfile.smart_open(fname, 'r') as fid:
        lines = fid.read().strip().splitlines()
    pairs = [ (os.path.join(root,l.split()[0]), os.path.join(root,l.split()[1])) for l in lines]
    return pairs
    
def load_pairs_from_list_file(fname, root=''):
    assert megfile.smart_isfile(fname), "cannot parse pairs from {:s}, file does not exist".format(fname)
    with megfile.smart_open(fname, 'r') as fid:
        lines = fid.read().strip().splitlines()
    pairs = [ (os.path.join(root,l+'_1.jpg'), os.path.join(root,l+'_2.jpg')) for l in lines if not l.startswith('#')]
    return pairs

def dsname_to_image_pairs(dnames, data_dir='./data/'):
    """
    dnames: list of datasets with image pairs, separated by +
    """
    all_pairs = []
    for dname in dnames.split('+'):
        if dname=='habitat_release':
            dirname = os.path.join(data_dir, 'habitat_release')
            assert megfile.smart_isdir(dirname), "cannot find folder for habitat_release pairs: "+dirname
            cache_file = os.path.join(dirname, 'pairs.txt')
            assert megfile.smart_isfile(cache_file), "cannot find cache file for habitat_release pairs, please first create the cache file, see instructions. "+cache_file
            pairs = load_pairs_from_cache_file(cache_file, root=dirname)
        elif dname in ['ARKitScenes', 'MegaDepth', '3DStreetView', 'IndoorVL']:
            dirname = os.path.join(data_dir, dname+'_crops')
            assert megfile.smart_isdir(dirname), "cannot find folder for {:s} pairs: {:s}".format(dname, dirname)
            list_file = os.path.join(dirname, 'listing.txt')
            assert megfile.smart_isfile(list_file), "cannot find list file for {:s} pairs, see instructions. {:s}".format(dname, list_file)
            pairs = load_pairs_from_list_file(list_file, root=dirname)
        print('  {:s}: {:,} pairs'.format(dname, len(pairs)))
        all_pairs += pairs 
    if '+' in dnames: print(' Total: {:,} pairs'.format(len(all_pairs)))
    return all_pairs 