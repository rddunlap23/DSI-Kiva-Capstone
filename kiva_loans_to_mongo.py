import pymongo
import os
import logging
import simplejson as json

from pymongo import MongoClient


#Setting up script logging
"""https://docs.python.org/2/library/logging.html"""

LEVEL = logging.DEBUG
FORMAT = logging.Formatter('%(asctime)-15s Line %(lineno)s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)
log.setLevel(LEVEL)
fhandler = logging.FileHandler('jsons_to_mongo.log')
shandler = logging.StreamHandler()
shandler.setLevel(logging.ERROR)
fhandler.setFormatter(FORMAT)
shandler.setFormatter(FORMAT)
log.addHandler(fhandler)
log.addHandler(shandler)

log.debug('Starting Kiva JSON to Mongo Script')


class connect_to_mongo():

    def __init__(self):
        try:
            self.client = MongoClient()        
            self.db = self.client.kivadb
        except:
            log.error('Could not establish a connection to Mongo Client')


def load_mongo(json_file, collection):
    #print "This function was called"
    mongo_collection = kivadb[collection]
    try:
        with open(json_file,'r') as json_data:
            including_header = json.load(json_data)
            data_dics = including_header[collection] #removes header and strips out good bits
            for d in data_dics:
                try:
                    mongo_collection.insert_one(d)
                except:
                    log.info('Could not insert the dict')
    except:
        log.error('Could not load json file: %s' %(json_file))
    
    return
    

def process_kiva_data(root_folder):
    """ Function will go through all three folders of kiva data downloaded from snapshots
    and pass these to a function to load to local MongoDB."""
    
    mongo_collections = {'loans':[],'lenders':[]}
    
    for collection in mongo_collections:
        path = os.path.join(root_folder,collection)
        log.debug('Getting files from %s' % path)
        
        try:
            file_names = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
            mongo_collections[collection] += file_names
            log.debug('Got %i JSON files in the %s collection' %(len(mongo_collections[collection]),collection))
        except:
            log.error('Could not get files at location. File may not exist.')
            
    #Write files to mongo for each of three collections
    counter = 0
    for collection, files in mongo_collections.iteritems(): 
        log.debug('Writing %s files to mongo now' %(collection))
        for f in files:
            counter += 1
            if counter%100:
                log.debug('Loading file:%s' %(f))
            load_mongo(f, collection) 


if __name__ == '__main__':
    mongo_conn = connect_to_mongo()
    kivadb = mongo_conn.db

    root_folder = '/Users/ryandunlap/kiva_capstone/json_snapshot/kiva_ds_json'
    process_kiva_data(root_folder)
    
    kivadb['loans'].create_index('id') #Index per ID provided by Kiva
