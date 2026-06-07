//! Resource allocation via attention mechanisms with auctions, decay, and scarcity.
//!
//! Models attention as a finite resource that agents compete for through
//! second-price auctions, with time-based decay and urgency scheduling.

use std::collections::HashMap;

// ── Module: budget ───────────────────────────────────────────────────────

/// A finite attention budget that can be allocated and tracked.
#[derive(Debug, Clone)]
pub struct AttentionBudget {
    total: f64,
    allocated: f64,
    allocations: HashMap<String, f64>,
}

impl AttentionBudget {
    /// Create a new budget with the given total capacity.
    pub fn new(total: f64) -> Self {
        assert!(total >= 0.0, "Budget must be non-negative");
        Self {
            total,
            allocated: 0.0,
            allocations: HashMap::new(),
        }
    }

    /// Allocate attention to a task. Returns true if successful.
    pub fn allocate(&mut self, task: &str, amount: f64) -> bool {
        if amount < 0.0 || self.allocated + amount > self.total {
            return false;
        }
        if let Some(existing) = self.allocations.get(task) {
            self.allocated -= existing;
        }
        self.allocated += amount;
        self.allocations.insert(task.to_string(), amount);
        true
    }

    /// Deallocate attention from a task.
    pub fn deallocate(&mut self, task: &str) -> Option<f64> {
        if let Some(amount) = self.allocations.remove(task) {
            self.allocated -= amount;
            Some(amount)
        } else {
            None
        }
    }

    /// Remaining budget.
    pub fn remaining(&self) -> f64 {
        self.total - self.allocated
    }

    /// Total budget capacity.
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Currently allocated amount.
    pub fn allocated(&self) -> f64 {
        self.allocated
    }

    /// Get allocation for a specific task.
    pub fn get_allocation(&self, task: &str) -> Option<f64> {
        self.allocations.get(task).copied()
    }

    /// Number of active allocations.
    pub fn num_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            self.allocated / self.total
        }
    }

    /// Resize total budget, deallocating if necessary.
    pub fn resize(&mut self, new_total: f64) -> f64 {
        let _old = self.total;
        self.total = new_total;
        if self.allocated > self.total {
            let excess = self.allocated - self.total;
            self.allocated = self.total;
            excess
        } else {
            0.0
        }
    }

    /// List all tasks with their allocations.
    pub fn allocations(&self) -> &HashMap<String, f64> {
        &self.allocations
    }
}

// ── Module: auction ──────────────────────────────────────────────────────

/// A bid in a second-price auction.
#[derive(Debug, Clone)]
pub struct Bid {
    pub bidder: String,
    pub amount: f64,
}

/// Result of a second-price auction.
#[derive(Debug, Clone)]
pub struct AuctionResult {
    pub winner: String,
    pub price: f64,
    pub bids: Vec<Bid>,
}

/// Run a second-price sealed-bid auction.
/// Winner pays the second-highest bid price.
pub fn second_price_auction(bids: Vec<Bid>) -> Option<AuctionResult> {
    if bids.is_empty() {
        return None;
    }
    let mut sorted = bids.clone();
    sorted.sort_by(|a, b| b.amount.partial_cmp(&a.amount).unwrap());

    let winner = sorted[0].clone();
    let price = if sorted.len() > 1 {
        sorted[1].amount
    } else {
        winner.amount
    };

    Some(AuctionResult {
        winner: winner.bidder,
        price,
        bids,
    })
}

/// Run a Vickrey auction (second-price with reserve).
pub fn vickrey_auction(bids: Vec<Bid>, reserve: f64) -> Option<AuctionResult> {
    let above_reserve: Vec<Bid> = bids.into_iter().filter(|b| b.amount >= reserve).collect();
    if above_reserve.is_empty() {
        return None;
    }
    if above_reserve.len() == 1 {
        // Single bid above reserve: winner pays the reserve price
        let _winner_bid = above_reserve[0].amount;
        let winner_name = above_reserve[0].bidder.clone();
        return Some(AuctionResult {
            winner: winner_name,
            price: reserve,
            bids: above_reserve,
        });
    }
    let mut result = second_price_auction(above_reserve)?;
    result.price = result.price.max(reserve);
    Some(result)
}

/// Compute the revenue for a series of auctions.
pub fn auction_revenue(results: &[AuctionResult]) -> f64 {
    results.iter().map(|r| r.price).sum()
}

// ── Module: decay ────────────────────────────────────────────────────────

/// Exponential decay model for attention.
#[derive(Debug, Clone)]
pub struct AttentionDecay {
    pub half_life: f64,
    pub initial_value: f64,
    pub elapsed: f64,
}

impl AttentionDecay {
    /// Create a new decay model.
    pub fn new(half_life: f64, initial_value: f64) -> Self {
        Self {
            half_life,
            initial_value,
            elapsed: 0.0,
        }
    }

    /// Current value after decay.
    pub fn current_value(&self) -> f64 {
        if self.half_life <= 0.0 {
            return self.initial_value;
        }
        let decay_rate = (2.0_f64).ln() / self.half_life;
        self.initial_value * (-decay_rate * self.elapsed).exp()
    }

    /// Advance time by dt.
    pub fn advance(&mut self, dt: f64) -> f64 {
        self.elapsed += dt;
        self.current_value()
    }

    /// Reset to initial value.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }

    /// Time until value drops below threshold.
    pub fn time_to_threshold(&self, threshold: f64) -> Option<f64> {
        if threshold >= self.initial_value {
            return Some(0.0);
        }
        if self.half_life <= 0.0 {
            return None;
        }
        let decay_rate = (2.0_f64).ln() / self.half_life;
        let t = -(threshold / self.initial_value).ln() / decay_rate;
        Some(t - self.elapsed)
    }

    /// Is the attention effectively zero (< 0.01)?
    pub fn is_dead(&self) -> bool {
        self.current_value() < 0.01
    }
}

/// Apply decay to all allocations in a budget.
pub fn decay_budget(budget: &mut AttentionBudget, decay: &AttentionDecay, dt: f64) {
    let factor = {
        let d = AttentionDecay::new(decay.half_life, 1.0);
        let decay_rate = (2.0_f64).ln() / d.half_life;
        (-decay_rate * dt).exp()
    };
    let tasks: Vec<String> = budget.allocations().keys().cloned().collect();
    for task in tasks {
        if let Some(amount) = budget.get_allocation(&task) {
            let new_amount = amount * factor;
            budget.deallocate(&task);
            if new_amount > 0.01 {
                budget.allocate(&task, new_amount);
            }
        }
    }
}

// ── Module: priority_queue ───────────────────────────────────────────────

/// Priority with urgency factor.
#[derive(Debug, Clone)]
pub struct PriorityItem {
    pub id: String,
    pub base_priority: f64,
    pub urgency: f64,
    pub deadline: Option<f64>,
    pub created_at: f64,
}

impl PriorityItem {
    /// Effective priority = base * urgency * time_factor.
    pub fn effective_priority(&self, current_time: f64) -> f64 {
        let time_factor = match self.deadline {
            Some(dl) if dl > current_time => 1.0 / (dl - current_time + 1.0),
            Some(_) => 10.0, // overdue, maximum urgency
            None => 1.0,
        };
        self.base_priority * self.urgency * time_factor
    }
}

/// Urgency-based priority queue.
#[derive(Debug, Clone)]
pub struct PriorityQueue {
    items: Vec<PriorityItem>,
}

impl PriorityQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Enqueue an item.
    pub fn enqueue(&mut self, item: PriorityItem) {
        self.items.push(item);
    }

    /// Dequeue the highest-priority item at the given time.
    pub fn dequeue(&mut self, current_time: f64) -> Option<PriorityItem> {
        if self.items.is_empty() {
            return None;
        }
        let best_idx = self
            .items
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.effective_priority(current_time)
                    .partial_cmp(&b.effective_priority(current_time))
                    .unwrap()
            })
            .map(|(i, _)| i)?;
        Some(self.items.remove(best_idx))
    }

    /// Peek at the highest-priority item without removing it.
    pub fn peek(&self, current_time: f64) -> Option<&PriorityItem> {
        self.items
            .iter()
            .max_by(|a, b| {
                a.effective_priority(current_time)
                    .partial_cmp(&b.effective_priority(current_time))
                    .unwrap()
            })
    }

    /// Number of items in the queue.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Is the queue empty?
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Remove an item by id.
    pub fn remove(&mut self, id: &str) -> Option<PriorityItem> {
        if let Some(idx) = self.items.iter().position(|i| i.id == id) {
            Some(self.items.remove(idx))
        } else {
            None
        }
    }

    /// Get sorted order at a given time.
    pub fn sorted(&self, current_time: f64) -> Vec<&PriorityItem> {
        let mut refs: Vec<&PriorityItem> = self.items.iter().collect();
        refs.sort_by(|a, b| {
            b.effective_priority(current_time)
                .partial_cmp(&a.effective_priority(current_time))
                .unwrap()
        });
        refs
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ── Module: scarcity ─────────────────────────────────────────────────────

/// Resource contention simulation.
#[derive(Debug, Clone)]
pub struct ScarceResource {
    pub name: String,
    pub capacity: f64,
    pub consumers: HashMap<String, f64>,
}

impl ScarceResource {
    /// Create a new scarce resource.
    pub fn new(name: &str, capacity: f64) -> Self {
        Self {
            name: name.to_string(),
            capacity,
            consumers: HashMap::new(),
        }
    }

    /// Available capacity.
    pub fn available(&self) -> f64 {
        self.capacity - self.consumers.values().sum::<f64>()
    }

    /// Try to consume some amount. Returns true if successful.
    pub fn consume(&mut self, consumer: &str, amount: f64) -> bool {
        if amount <= self.available() {
            *self.consumers.entry(consumer.to_string()).or_insert(0.0) += amount;
            true
        } else {
            false
        }
    }

    /// Release consumption.
    pub fn release(&mut self, consumer: &str) -> Option<f64> {
        self.consumers.remove(consumer)
    }

    /// Number of consumers.
    pub fn num_consumers(&self) -> usize {
        self.consumers.len()
    }

    /// Contention ratio (0.0 = no contention, 1.0 = fully consumed).
    pub fn contention(&self) -> f64 {
        if self.capacity == 0.0 {
            0.0
        } else {
            self.consumers.values().sum::<f64>() / self.capacity
        }
    }
}

/// Multi-resource contention.
#[derive(Debug, Clone)]
pub struct ResourceArena {
    resources: HashMap<String, ScarceResource>,
}

impl ResourceArena {
    /// Create a new arena.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }

    /// Add a resource.
    pub fn add_resource(&mut self, resource: ScarceResource) {
        self.resources.insert(resource.name.clone(), resource);
    }

    /// Try to acquire multiple resources for a consumer.
    pub fn acquire(&mut self, consumer: &str, demands: &HashMap<String, f64>) -> bool {
        // Check all available first
        for (name, amount) in demands {
            if let Some(res) = self.resources.get(name) {
                if res.available() < *amount {
                    return false;
                }
            } else {
                return false;
            }
        }
        // All available, consume
        for (name, amount) in demands {
            self.resources.get_mut(name).unwrap().consume(consumer, *amount);
        }
        true
    }

    /// Release all resources for a consumer.
    pub fn release_all(&mut self, consumer: &str) {
        for res in self.resources.values_mut() {
            res.release(consumer);
        }
    }

    /// Get a resource by name.
    pub fn get_resource(&self, name: &str) -> Option<&ScarceResource> {
        self.resources.get(name)
    }

    /// Total number of resources.
    pub fn num_resources(&self) -> usize {
        self.resources.len()
    }
}

impl Default for ResourceArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Budget tests ──

    #[test]
    fn test_budget_new() {
        let b = AttentionBudget::new(100.0);
        assert_eq!(b.total(), 100.0);
        assert_eq!(b.remaining(), 100.0);
        assert_eq!(b.allocated(), 0.0);
    }

    #[test]
    fn test_budget_allocate() {
        let mut b = AttentionBudget::new(100.0);
        assert!(b.allocate("task1", 30.0));
        assert_eq!(b.remaining(), 70.0);
        assert_eq!(b.allocated(), 30.0);
    }

    #[test]
    fn test_budget_over_allocate() {
        let mut b = AttentionBudget::new(100.0);
        assert!(b.allocate("task1", 60.0));
        assert!(!b.allocate("task2", 50.0)); // not enough
    }

    #[test]
    fn test_budget_deallocate() {
        let mut b = AttentionBudget::new(100.0);
        b.allocate("task1", 40.0);
        assert_eq!(b.deallocate("task1"), Some(40.0));
        assert_eq!(b.remaining(), 100.0);
    }

    #[test]
    fn test_budget_utilization() {
        let mut b = AttentionBudget::new(200.0);
        b.allocate("t", 50.0);
        assert!((b.utilization() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_budget_reallocate() {
        let mut b = AttentionBudget::new(100.0);
        b.allocate("task1", 50.0);
        b.allocate("task1", 30.0); // replace
        assert_eq!(b.get_allocation("task1"), Some(30.0));
        assert_eq!(b.allocated(), 30.0);
    }

    #[test]
    fn test_budget_negative_alloc() {
        let mut b = AttentionBudget::new(100.0);
        assert!(!b.allocate("t", -10.0));
    }

    #[test]
    fn test_budget_resize() {
        let mut b = AttentionBudget::new(100.0);
        b.allocate("t", 60.0);
        let excess = b.resize(50.0);
        assert!(excess > 0.0);
    }

    #[test]
    fn test_budget_resize_larger() {
        let mut b = AttentionBudget::new(100.0);
        b.allocate("t", 50.0);
        let excess = b.resize(200.0);
        assert_eq!(excess, 0.0);
        assert_eq!(b.total(), 200.0);
    }

    // ── Auction tests ──

    #[test]
    fn test_auction_single_bid() {
        let result = second_price_auction(vec![Bid {
            bidder: "a".into(),
            amount: 10.0,
        }]);
        let r = result.unwrap();
        assert_eq!(r.winner, "a");
        assert_eq!(r.price, 10.0); // own bid when alone
    }

    #[test]
    fn test_auction_two_bids() {
        let result = second_price_auction(vec![
            Bid { bidder: "a".into(), amount: 100.0 },
            Bid { bidder: "b".into(), amount: 80.0 },
        ]);
        let r = result.unwrap();
        assert_eq!(r.winner, "a");
        assert_eq!(r.price, 80.0); // second price
    }

    #[test]
    fn test_auction_multiple_bids() {
        let result = second_price_auction(vec![
            Bid { bidder: "a".into(), amount: 50.0 },
            Bid { bidder: "b".into(), amount: 100.0 },
            Bid { bidder: "c".into(), amount: 75.0 },
        ]);
        let r = result.unwrap();
        assert_eq!(r.winner, "b");
        assert_eq!(r.price, 75.0);
    }

    #[test]
    fn test_auction_empty() {
        assert!(second_price_auction(vec![]).is_none());
    }

    #[test]
    fn test_vickrey_with_reserve() {
        let result = vickrey_auction(
            vec![
                Bid { bidder: "a".into(), amount: 50.0 },
                Bid { bidder: "b".into(), amount: 80.0 },
            ],
            60.0,
        );
        let r = result.unwrap();
        assert_eq!(r.winner, "b");
        assert_eq!(r.price, 60.0); // max(second_price, reserve)
    }

    #[test]
    fn test_vickrey_all_below_reserve() {
        let result = vickrey_auction(
            vec![Bid { bidder: "a".into(), amount: 10.0 }],
            50.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_auction_revenue() {
        let r1 = AuctionResult { winner: "a".into(), price: 30.0, bids: vec![] };
        let r2 = AuctionResult { winner: "b".into(), price: 50.0, bids: vec![] };
        assert_eq!(auction_revenue(&[r1, r2]), 80.0);
    }

    // ── Decay tests ──

    #[test]
    fn test_decay_initial() {
        let d = AttentionDecay::new(10.0, 100.0);
        assert!((d.current_value() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_half_life() {
        let d = AttentionDecay::new(10.0, 100.0);
        let mut d = d;
        d.advance(10.0);
        assert!((d.current_value() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_decay_zero_half_life() {
        let d = AttentionDecay::new(0.0, 100.0);
        assert!((d.current_value() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_advance() {
        let mut d = AttentionDecay::new(5.0, 80.0);
        d.advance(1.0);
        assert!(d.current_value() < 80.0);
        assert!(d.current_value() > 0.0);
    }

    #[test]
    fn test_decay_reset() {
        let mut d = AttentionDecay::new(5.0, 100.0);
        d.advance(20.0);
        d.reset();
        assert!((d.current_value() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_time_to_threshold() {
        let d = AttentionDecay::new(10.0, 100.0);
        let t = d.time_to_threshold(50.0).unwrap();
        assert!((t - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_decay_is_dead() {
        let mut d = AttentionDecay::new(1.0, 1.0);
        d.advance(100.0);
        assert!(d.is_dead());
    }

    #[test]
    fn test_decay_not_dead() {
        let d = AttentionDecay::new(100.0, 100.0);
        assert!(!d.is_dead());
    }

    #[test]
    fn test_decay_budget() {
        let mut budget = AttentionBudget::new(100.0);
        budget.allocate("t1", 50.0);
        let decay = AttentionDecay::new(10.0, 1.0);
        decay_budget(&mut budget, &decay, 10.0);
        let alloc = budget.get_allocation("t1").unwrap();
        assert!(alloc < 50.0);
    }

    // ── Priority Queue tests ──

    #[test]
    fn test_pq_new() {
        let pq = PriorityQueue::new();
        assert!(pq.is_empty());
    }

    #[test]
    fn test_pq_enqueue_dequeue() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "a".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        assert_eq!(pq.len(), 1);
        let item = pq.dequeue(0.0).unwrap();
        assert_eq!(item.id, "a");
        assert!(pq.is_empty());
    }

    #[test]
    fn test_pq_priority_ordering() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "low".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        pq.enqueue(PriorityItem {
            id: "high".into(),
            base_priority: 10.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        let first = pq.dequeue(0.0).unwrap();
        assert_eq!(first.id, "high");
    }

    #[test]
    fn test_pq_urgency_factor() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "a".into(),
            base_priority: 5.0,
            urgency: 2.0,
            deadline: None,
            created_at: 0.0,
        });
        pq.enqueue(PriorityItem {
            id: "b".into(),
            base_priority: 5.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        let first = pq.dequeue(0.0).unwrap();
        assert_eq!(first.id, "a");
    }

    #[test]
    fn test_pq_deadline() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "urgent".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: Some(1.0),
            created_at: 0.0,
        });
        pq.enqueue(PriorityItem {
            id: "relaxed".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: Some(100.0),
            created_at: 0.0,
        });
        let first = pq.dequeue(0.0).unwrap();
        assert_eq!(first.id, "urgent");
    }

    #[test]
    fn test_pq_overdue() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "overdue".into(),
            base_priority: 0.1,
            urgency: 1.0,
            deadline: Some(5.0),
            created_at: 0.0,
        });
        let item = pq.dequeue(10.0).unwrap();
        assert_eq!(item.effective_priority(10.0), 1.0); // 10.0 * urgency
    }

    #[test]
    fn test_pq_peek() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "x".into(),
            base_priority: 5.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        assert_eq!(pq.peek(0.0).unwrap().id, "x");
        assert_eq!(pq.len(), 1);
    }

    #[test]
    fn test_pq_remove() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "a".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        let removed = pq.remove("a").unwrap();
        assert_eq!(removed.id, "a");
        assert!(pq.is_empty());
    }

    #[test]
    fn test_pq_sorted() {
        let mut pq = PriorityQueue::new();
        pq.enqueue(PriorityItem {
            id: "c".into(),
            base_priority: 1.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        pq.enqueue(PriorityItem {
            id: "a".into(),
            base_priority: 10.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        pq.enqueue(PriorityItem {
            id: "b".into(),
            base_priority: 5.0,
            urgency: 1.0,
            deadline: None,
            created_at: 0.0,
        });
        let sorted: Vec<&str> = pq.sorted(0.0).iter().map(|i| i.id.as_str()).collect();
        assert_eq!(sorted, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_pq_default() {
        let pq = PriorityQueue::default();
        assert!(pq.is_empty());
    }

    // ── Scarcity tests ──

    #[test]
    fn test_scarce_new() {
        let r = ScarceResource::new("cpu", 100.0);
        assert_eq!(r.available(), 100.0);
        assert_eq!(r.contention(), 0.0);
    }

    #[test]
    fn test_scarce_consume() {
        let mut r = ScarceResource::new("cpu", 100.0);
        assert!(r.consume("agent1", 30.0));
        assert_eq!(r.available(), 70.0);
    }

    #[test]
    fn test_scarce_over_consume() {
        let mut r = ScarceResource::new("cpu", 100.0);
        r.consume("a", 80.0);
        assert!(!r.consume("b", 30.0));
    }

    #[test]
    fn test_scarce_release() {
        let mut r = ScarceResource::new("cpu", 100.0);
        r.consume("a", 50.0);
        assert_eq!(r.release("a"), Some(50.0));
        assert_eq!(r.available(), 100.0);
    }

    #[test]
    fn test_scarce_contention() {
        let mut r = ScarceResource::new("cpu", 100.0);
        r.consume("a", 75.0);
        assert!((r.contention() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_scarce_multiple_consumers() {
        let mut r = ScarceResource::new("mem", 200.0);
        r.consume("a", 50.0);
        r.consume("b", 70.0);
        assert_eq!(r.num_consumers(), 2);
        assert_eq!(r.available(), 80.0);
    }

    #[test]
    fn test_arena_new() {
        let a = ResourceArena::new();
        assert_eq!(a.num_resources(), 0);
    }

    #[test]
    fn test_arena_acquire() {
        let mut a = ResourceArena::new();
        a.add_resource(ScarceResource::new("cpu", 100.0));
        a.add_resource(ScarceResource::new("mem", 200.0));
        let mut demands = HashMap::new();
        demands.insert("cpu".into(), 50.0);
        demands.insert("mem".into(), 100.0);
        assert!(a.acquire("agent1", &demands));
    }

    #[test]
    fn test_arena_acquire_fails() {
        let mut a = ResourceArena::new();
        a.add_resource(ScarceResource::new("cpu", 100.0));
        let mut demands = HashMap::new();
        demands.insert("cpu".into(), 150.0);
        assert!(!a.acquire("agent1", &demands));
    }

    #[test]
    fn test_arena_release_all() {
        let mut a = ResourceArena::new();
        a.add_resource(ScarceResource::new("cpu", 100.0));
        let mut demands = HashMap::new();
        demands.insert("cpu".into(), 50.0);
        a.acquire("agent1", &demands);
        a.release_all("agent1");
        assert_eq!(a.get_resource("cpu").unwrap().available(), 100.0);
    }

    #[test]
    fn test_arena_missing_resource() {
        let mut a = ResourceArena::new();
        a.add_resource(ScarceResource::new("cpu", 100.0));
        let mut demands = HashMap::new();
        demands.insert("gpu".into(), 50.0);
        assert!(!a.acquire("agent1", &demands));
    }

    #[test]
    fn test_arena_default() {
        let a = ResourceArena::default();
        assert_eq!(a.num_resources(), 0);
    }

    #[test]
    fn test_budget_zero() {
        let mut b = AttentionBudget::new(0.0);
        assert!(!b.allocate("t", 1.0));
        assert_eq!(b.utilization(), 0.0);
    }

    #[test]
    fn test_scarce_zero_capacity() {
        let mut r = ScarceResource::new("x", 0.0);
        assert_eq!(r.contention(), 0.0);
        assert!(!r.consume("a", 1.0));
    }

    #[test]
    fn test_auction_tie_breaking() {
        let result = second_price_auction(vec![
            Bid { bidder: "a".into(), amount: 100.0 },
            Bid { bidder: "b".into(), amount: 100.0 },
        ]);
        let r = result.unwrap();
        assert_eq!(r.price, 100.0); // second price = same as first
    }

    #[test]
    fn test_decay_multiple_steps() {
        let mut d = AttentionDecay::new(5.0, 100.0);
        for _ in 0..5 {
            d.advance(1.0);
        }
        assert!(d.current_value() < 100.0);
        assert!(d.current_value() > 0.0);
    }

    #[test]
    fn test_pq_empty_dequeue() {
        let mut pq = PriorityQueue::new();
        assert!(pq.dequeue(0.0).is_none());
    }

    #[test]
    fn test_budget_deallocate_nonexistent() {
        let mut b = AttentionBudget::new(100.0);
        assert_eq!(b.deallocate("nope"), None);
    }

    #[test]
    fn test_decay_time_to_threshold_immediate() {
        let d = AttentionDecay::new(10.0, 100.0);
        let t = d.time_to_threshold(100.0).unwrap();
        assert!((t - 0.0).abs() < 1e-10);
    }
}
