# Evaluation Quiz Answer Key

These answers use the quiz feedback from the attempts recorded in `2nd try.md`. The Python LensKit run in this repository is an approximation of the original Java evaluation, so use the quiz feedback where its results differ.

1. **Why do Pers-Mean and Item-Mean have the same nDCG?**  
   The user mean is constant across all items for a user and has no influence on their relative order.

2. **Which algorithm has the best RMSE?**  
   **Item-Item**

3. **Which algorithm has the best MAP?**  
   **Popular**

4. **Which algorithms beat the personalized mean on RMSE? (Select all that apply.)**  
   **Item-Item**, **Normalized User-User**, **Cosine User-User**, and **Normalized Lucene**.

5. **What is the best neighborhood size for user-user to produce accurate predictions and rankings?**  
   **40**

   If your quiz attempt instead asks about **item-item**, try **20**.

6. **What is the best personalized algorithm family for top-N recommendations on this data?**  
   **User-User**

7. **Do the top-N metrics generally agree or disagree?**  
   **Agree**

8. **As the neighborhood size increases, what happens to user-user's top-N accuracy?**  
   **Improves**

9. **As the neighborhood size increases, what happens to Lucene's top-N accuracy?**  
   **Gets worse**

10. **As the neighborhood size increases, what happens to item-item's top-N accuracy?**  
    **Plateaus**

11. **Which two algorithms produce the most diverse recommendations?**  
    **Popular** and **User-User**

12. **Does increasing the neighborhood size make Lucene provide more or less diverse recommendations?**  
    **Less diverse**

13. **What is the top-N nDCG of Popular?**  
    **0.08**

14. **What is the best algorithm to deploy for top-N recommendation?**  
    **Popular**
