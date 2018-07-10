function makePromise(url) {
       // Sets up a promise in the proper way using webix
       return new webix.promise(function(success, fail) {
           webix.ajax(url, function(text){
               if (text) success (text);
               else fail(text.error)
           })
       })
   }

   function girderHelpers(requestType, girderObjectID = null) {
       switch (requestType) {
           case 'getCollURL':
               url = config.BASE_URL + "/resource/lookup?path=collection/" + config.COLLECTION_NAME;
               promise = makePromise(url);
               break;
           case 'listFoldersInCollection':
               url = config.BASE_URL + "/folder?limit=1000&parentType=collection&parentId=" + girderObjectID;
               promise = makePromise(url);   
               break;
           case 'listFoldersinFolder':
               url = config.BASE_URL + "/folder?parentType=folder&parentId=" + girderObjectID;
               promise = makePromise(url);
               break;
               //adrc.digitalslidearchive.emory.edu:8080/api/v1/item?folderId=5ad11d6a92ca9a001adee5b3&limit=50&sort=lowerName&sortdir=1

           case 'listItemsInFolder':
               url = config.BASE_URL + "/item?folderId=" + girderObjectID + "&limit=5000"
               // url = config.BASE_URL + "/item?limit=500&folderId=" + girderObjectID;
               promise = makePromise(url);
               break;
           default:
               console.log("No case found.....errors will happen");
       }
       return promise;
   }




