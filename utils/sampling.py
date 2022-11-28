import numpy as np

def noniid(dataset, args):

    idxs = np.arange(len(dataset))
    labels = np.transpose(np.array(dataset.labels))
    
    dict_users = {i: list() for i in range(args.num_clients)}
    dict_labels = dict()
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = list(idxs_labels[0])
    labels = idxs_labels[1]
    
    if args.quantity_skew:
        min_num, max_num = 500, 1000
        num_rand = np.random.randint(min_num, max_num+1, size=args.num_clients)

    if args.alpha > 0:
        proportions = np.random.dirichlet(np.ones(args.num_classes) * args.alpha, args.num_clients)
    else:
        rand_class_num = np.random.randint(0, 10, size=args.num_clients)
        
    for i in range(args.num_classes):
        specific_class = set(np.extract(labels == i, idxs))
        dict_labels.update({i : specific_class})

    if args.alpha > 0:
        for i, prop in enumerate(proportions):
            
            if args.quantity_skew:
                prop = num_rand[i] * prop
            else:
                prop = args.num_data * prop
                
            rand_set = list()
            for c in range(args.num_classes):
                try:
                    rand_class = list(np.random.choice(list(dict_labels[c]), int(prop[c])))
                    dict_labels[c] = dict_labels[c] - set(rand_class)
                    rand_set = rand_set + rand_class
                except ValueError as v:
                    pass
            dict_users[i] = set(rand_set)
    else:
        rand_set = list()
    
        for i, class_num in enumerate(rand_class_num):
            rand_set = list()
            if args.quantity_skew:
                rand_class = list(np.random.choice(list(dict_labels[class_num]), num_rand[i]))
            else:
                rand_class = list(np.random.choice(list(dict_labels[class_num]), args.num_data))
            dict_labels[class_num] = dict_labels[class_num] - set(rand_class)
            
            rand_set = rand_set + rand_class
            dict_users[i] = set(rand_set)
    
    return dict_users