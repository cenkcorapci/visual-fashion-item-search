db.products.find(
   { deepFeatures :
       { $near :
          {
            $geometry : {
               type : "Point" ,
               coordinates : [1.5, 0.9] },
            $maxDistance : 5
          }
       }
    }
).limit(3)