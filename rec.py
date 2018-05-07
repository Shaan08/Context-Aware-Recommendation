# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:45:56 2018

@author: ezarpkm
"""

import numpy as np
import pandas as pd

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.productmanufacturerid = None
        self.category = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, productmanufacturerid, category):
        self.train_data = train_data
        self.productmanufacturerid = productmanufacturerid
        self.category = category
        


        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.category]).agg({self.productmanufacturerid: 'count'}).reset_index()
        train_data_grouped
        train_data_grouped.rename(columns = {'productmanufacturerid': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.category], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, productmanufacturerid):    
        product_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        product_recommendations['productmanufacturerid'] = productmanufacturerid
    
        #Bring user_id column to the front
        cols = product_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        product_recommendations = product_recommendations[cols]
        
        return product_recommendations
    
    


class item_similarity_recommendation():
    def __init__(self):
        self.train_data = None
        self.prodmanid = None
        self.cat = None
        self.cooccurence_matrix = None
        self.cat_dict = None
        self.revcat_dict = None
        self.item_sim_rec = None
        
        
        
        #get unique product for every product manafacturer id i.e:
        #products made by different manufacturers
    def get_manufacturer_items(self, productmanufacturer):
        prod_man_list = self.train_data[self.train_data[self.prodmanid] == productmanufacturer]
        products_for_man_id = list(prod_man_list[self.cat].unique())
        
        return products_for_man_id
        
    
        #get unique manufacturers for every product
    def get_manufacturers(self, category):
        cat_data = self.train_data[self.train_data[self.cat] == category]
        manufacturer_id = set(cat_data[self.prodmanid].unique())
            
        return manufacturer_id
    
    def get_unique_products(self):
        all_products = list(self.train_data[self.cat].unique())
        
        return all_products
    
    
    #construct cooccurence matrix
    
    def create_coccurence_matrix(self,filterproducts, allproducts):
        
        #get productmanufacturerid for all products in filterproducts
        
        filterproducts_prod = []
        for i in range(0, len(filterproducts)):
            filterproducts_prod.append(self.get_manufacturers(filterproducts[i]))
            
        ###############################################
        #Initialize the products cooccurence matrix of size 
        #len(filterproducts) X len(products)
        ################################   
        
        cooccurence_matrix = np.matrix(np.zeros(shape =(len(filterproducts), len(allproducts))),float)
        
        
        #find similarity between filterprofucts and all products in the training data
        for i in range(0,len(allproducts)):
            #calculate unique manufacturers for each category i
            category_i_data = self.train_data(self.train_data(self.cat) == allproducts[i])
            manufacturers_i = set(category_i_data[self.prodmanid].unique())
            
            
            for j in range(0, len(filterproducts)):
                #get unique manufcaturers for category j
                manufacturers_j = filterproducts_prod[j]
                
                #calculate intersection of manufacturers of category i and j
                manufactureres_intersection = manufacturers_i.intersection(manufacturers_j)
                
                
                #calculate cooccurence_matrix[i,j] as jaccard index
                if len(manufactureres_intersection) != 0:
                    #calcualte union of manufacturers of category i and j
                    manufacturers_union = manufacturers_i.union(manufacturers_j)
                    
                    
                    cooccurence_matrix[j,i] = float(len(manufactureres_intersection))/float(len(manufacturers_union))
                    
                else:
                    cooccurence_matrix[j,i] = 0
            
        return cooccurence_matrix
                
    #Using cooccurence matrix to generate receommendations:
    def generate_top_recommendations(self, manid, cooccurencematrix, allprod, filterprod):
        print ("Non zero values in cooccurence matrix :%d" % np.count_nonzero(cooccurencematrix))
    
    #calculate weighted average of the scores in cooccurence matrix for all manufacturers product
        man_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        man_sim_scores = np.array(man_sim_scores)[0].tolist()
        
    #sort the indices of man_sim_scores based upon their value while maintaining the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(man_sim_scores))), reverse=True)
        
        #create a dataframe using columns manufaturerid, cateogry, score, rank
        columns = ['Manufacturer Id' , 'Products' , 'Score' , 'Rank']
        
        df = pd.DataFrame(columns = columns)
        
        #Create a dataframe with top 10 recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and allprod[sort_index[i][1]] not in filterprod and rank <= 10:
                df.loc[len(df)]=[manid,allprod[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current manufacturer has no products for training the item similarity based recommendation model.")
            return -1
        else:
            return df
        
       
    #Item similarity model
    def create(self, traindata, manid, category):
        self.train_data = traindata
        self.prodmanid = manid
        self.cat = category
        print ("test test test")
     
     
        
        
        
        
       
                
        
        
        
        
        
        #getting similar products for the given category
        
        
        
        
        
        
         #Use the Item similarity based receommendation system model to make recommendations
        
    def recommend(self, manuid):
            
            
            #get unique manufactures for all cateogry
        filterprod = self.get_manufacturer_items(manuid)
            
            
        print ("Number of unique categories for the manufacturer : %d" % len(filterprod))
            
            
            #get all unique products in the training data
        allprod = self.get_unique_products()
            
        print("no. of unique songs in the training set: %d" % len(allprod))
            
            #create a cooccurence matrix
        cooccurencematrix = self.create_coccurence_matrix(filterprod,allprod)
            
        #Use cooccurence for recommendations
        df_recommendations = self.generate_top_recommendations(manuid, cooccurencematrix, allprod, filterprod)
                
        return df_recommendations
            
            
    def get_similar_items(self, category_list):
        filterprod = category_list
        
        
        
        
        ######################################################
        #B. Get all unique categories in the training data
        ######################################################
        allprod = self.get_unique_products()
        
        print("no. of unique categories in the training set: %d" % len(allprod))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(filterprod) X len(allprod)
        ###############################################
        cooccurence_matrix = self.create_coccurence_matrix(filterprod,allprod)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        manuid = ""
        df_recommendations = self.generate_top_recommendations(manuid, cooccurencematrix, allprod, filterprod)
         
        return df_recommendations
        
        
        
        
        
        
        
        
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
