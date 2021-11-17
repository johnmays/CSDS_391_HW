import main
# Question 1e:

def output_1e():

    print("Unambiguously Versicolor:")
    main.test_simple_classifier(0)
    main.test_simple_classifier(10)
    main.test_simple_classifier(30)
    print("Unambiguously Virginica:")
    main.test_simple_classifier(50)
    main.test_simple_classifier(70)
    main.test_simple_classifier(90)
    print("Near the Decision Boundary:")
    main.test_simple_classifier(56)
    main.test_simple_classifier(69)
    main.test_simple_classifier(33)

    main.plot_select_iris_data_1e()


# output_1e()