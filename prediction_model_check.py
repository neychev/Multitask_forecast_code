def prediction_model_check(myX, myY, idx, _mode = 'lin_reg'):
    # DRAFT VERSION!
    #
    # This function trains the chosen regression model and computes weigth matrix W.
    # Time series supposed to be stationary
    #   input:  myX: ndarray [N, p];    object-features matrix;
    #           myY: ndarray [N, k];    answers matrix;
    #           idx: int;               idx of control row;
    #           _mode: str;             mode from the list above;
    #
    #   output: W: ndarray [];          weigths matrix; if 0 returned smth bad occured;
    #           std_dev: int;           l2 error of prediction using chosen model;

    checkX = myX[idx, :]
    checkY = myY[idx, :]
    myX = np.delete(myX, idx, 0)
    myY = np.delete(myY, idx, 0)
    predY = checkY
    W = 0
    if _mode == 'lin_reg':
        W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(myX), myX)), np.transpose(myX)), myY)
        predY = np.dot(checkX, W)
    if _mode == 'Elastic_net':
        lin_mod = skl.MultiTaskElasticNet(max_iter=5000)
        lin_mod.fit(X = myX, y = myY)
        predY = lin_mod.predict(X = checkX.reshape(1, -1))
        W = lin_mod.coef_
    if _mode == 'Lasso':
        lin_mod = skl.MultiTaskLasso(max_iter=5000)
        lin_mod.fit(X = myX, y = myY)
        predY = lin_mod.predict(X = checkX.reshape(1, -1))
        W = lin_mod.coef_
    if _mode == 'LARS':
        lin_mod = skl.Lars()
        lin_mod.fit(X = myX, y = myY)
        predY = lin_mod.predict(X = checkX.reshape(1, -1))
        W = np.transpose(lin_mod.coef_)
    if _mode == 'stop':
        return -1
    std_dev = np.linalg.norm(predY - checkY)
    result_dict = {'std_dev': std_dev, 'W': W}
    return result_dict