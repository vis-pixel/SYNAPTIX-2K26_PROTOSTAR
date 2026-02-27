/**
 * BioRhythm X — API Service Layer
 * ────────────────────────────────
 * Central API client for all frontend ↔ FastAPI communication.
 * JWT auth, auto-refresh, polling, SSE, error handling.
 *
 * Usage:
 *   import { api, auth, vitals, predictions } from './api.js';
 */

// ─── Config ──────────────────────────────────────────────────────────────────
const BASE_URL = 'http://localhost:8000';
const API       = `${BASE_URL}/api`;

// ─── Token Manager ────────────────────────────────────────────────────────────
const TokenManager = {
  get accessToken()  { return localStorage.getItem('brx_access_token'); },
  get refreshToken() { return localStorage.getItem('brx_refresh_token'); },
  get user()         { return JSON.parse(localStorage.getItem('brx_user') || 'null'); },

  save(accessToken, refreshToken, user) {
    localStorage.setItem('brx_access_token', accessToken);
    if (refreshToken) localStorage.setItem('brx_refresh_token', refreshToken);
    if (user) localStorage.setItem('brx_user', JSON.stringify(user));
  },

  clear() {
    localStorage.removeItem('brx_access_token');
    localStorage.removeItem('brx_refresh_token');
    localStorage.removeItem('brx_user');
  },

  get isLoggedIn() {
    return !!this.accessToken;
  }
};

// ─── HTTP Client ──────────────────────────────────────────────────────────────
async function request(endpoint, options = {}) {
  const url = endpoint.startsWith('http') ? endpoint : `${API}${endpoint}`;

  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  // Attach JWT if available
  if (TokenManager.accessToken && !options.skipAuth) {
    headers['Authorization'] = `Bearer ${TokenManager.accessToken}`;
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    // 401 → try refresh token
    if (response.status === 401 && TokenManager.refreshToken && !options._retried) {
      const refreshed = await auth.refreshToken();
      if (refreshed) {
        return request(endpoint, { ...options, _retried: true });
      }
      // Refresh failed → logout
      auth.logout();
      window.location.href = 'login.html';
      throw new Error('Session expired');
    }

    // 429 → rate limited
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After') || 5;
      console.warn(`Rate limited. Retry after ${retryAfter}s`);
      throw new ApiError('Rate limited', 429, { retryAfter: Number(retryAfter) });
    }

    // Parse response
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new ApiError(
        data?.detail || `HTTP ${response.status}`,
        response.status,
        data
      );
    }

    return data;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network error', 0, null);
  }
}

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
    this.name = 'ApiError';
  }
}

// ─── Auth Service ─────────────────────────────────────────────────────────────
const auth = {
  async login(username, password) {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const res = await fetch(`${API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) throw new ApiError(data.detail || 'Login failed', res.status, data);

    TokenManager.save(data.access_token, data.refresh_token, data.user);
    return data;
  },

  async register(username, email, password) {
    return request('/auth/register', {
      method: 'POST',
      body: { username, email, password },
      skipAuth: true,
    });
  },

  async refreshToken() {
    try {
      const res = await fetch(`${API}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: TokenManager.refreshToken }),
      });
      if (!res.ok) return false;
      const data = await res.json();
      TokenManager.save(data.access_token, data.refresh_token);
      return true;
    } catch {
      return false;
    }
  },

  logout() {
    TokenManager.clear();
    sessionStorage.removeItem('biorhythm_logged_in');
    window.location.href = 'login.html';
  },

  get isLoggedIn() { return TokenManager.isLoggedIn; },
  get user()       { return TokenManager.user; },
  get token()      { return TokenManager.accessToken; },
};

// ─── Vitals Service ───────────────────────────────────────────────────────────
const vitals = {
  /** POST single vitals reading */
  async ingest(data) {
    return request('/vitals/ingest', { method: 'POST', body: data });
  },

  /** GET vitals history */
  async history(limit = 100) {
    return request(`/vitals/history?limit=${limit}`);
  },

  /** POST batch smartwatch data via device token */
  async deviceIngest(readings, deviceToken) {
    return request('/devices/ingest/batch', {
      method: 'POST',
      body: { readings },
      headers: { 'X-Device-Token': deviceToken },
      skipAuth: true,
    });
  },
};

// ─── Steps Service ────────────────────────────────────────────────────────────
const steps = {
  async analyze(accelX, accelY, accelZ, heightCm = 170) {
    return request('/steps/analyze', {
      method: 'POST',
      body: { accel_x: accelX, accel_y: accelY, accel_z: accelZ, height_cm: heightCm },
    });
  },
};

// ─── Calories Service ─────────────────────────────────────────────────────────
const calories = {
  async estimate(data) {
    return request('/calories/estimate', { method: 'POST', body: data });
  },
};

// ─── Predictions & Risk ───────────────────────────────────────────────────────
const predictions = {
  async run(vitalsData, profileData) {
    return request('/predictions/run', {
      method: 'POST',
      body: { vitals: vitalsData, profile: profileData },
    });
  },

  async riskScore(vitalsData, profileData) {
    return request('/risk/score', {
      method: 'POST',
      body: { vitals: vitalsData, profile: profileData },
    });
  },
};

// ─── Anomaly Service ──────────────────────────────────────────────────────────
const anomaly = {
  async detect(vitalsSnapshot) {
    return request('/anomaly/detect', { method: 'POST', body: vitalsSnapshot });
  },

  async logs(userId, limit = 50, severity = null) {
    let url = `/anomaly/logs/${userId}?limit=${limit}`;
    if (severity) url += `&severity=${severity}`;
    return request(url);
  },
};

// ─── Field Alerts Service ─────────────────────────────────────────────────────
const alerts = {
  async getAll(limit = 50) {
    return request(`/field-alerts?limit=${limit}`);
  },

  async acknowledge(alertId) {
    return request(`/field-alerts/${alertId}/ack`, { method: 'POST' });
  },
};

// ─── Water Tracker Service ────────────────────────────────────────────────────
const water = {
  async log(amountMl) {
    return request('/water/log', { method: 'POST', body: { amount_ml: amountMl } });
  },

  async today() {
    return request('/water/today');
  },

  async goal() {
    return request('/water/goal');
  },
};

// ─── Diet Service ─────────────────────────────────────────────────────────────
const diet = {
  async plan(profileData) {
    return request('/diet/plan', { method: 'POST', body: profileData });
  },

  async logMeal(mealData) {
    return request('/diet/meal', { method: 'POST', body: mealData });
  },

  async todayIntake() {
    return request('/diet/today');
  },
};

// ─── WhatsApp Service ─────────────────────────────────────────────────────────
const whatsapp = {
  async sendAlert(phone, alertData) {
    return request('/whatsapp/send', { method: 'POST', body: { phone, ...alertData } });
  },

  async status() {
    return request('/whatsapp/status');
  },
};

// ─── Dataset Service ──────────────────────────────────────────────────────────
const datasets = {
  async list() {
    return request('/datasets');
  },

  async status(name) {
    return request(`/datasets/${name}/status`);
  },
};

// ─── Devices Service ──────────────────────────────────────────────────────────
const devices = {
  async register(deviceData) {
    return request('/devices/register', { method: 'POST', body: deviceData });
  },

  async list() {
    return request('/devices');
  },
};

// ─── Health Check ─────────────────────────────────────────────────────────────
async function healthCheck() {
  return request(`${BASE_URL}/health`, { skipAuth: true });
}

// ─── Polling Manager ──────────────────────────────────────────────────────────
class Poller {
  constructor(fetchFn, intervalMs = 2500) {
    this._fetchFn = fetchFn;
    this._interval = intervalMs;
    this._timer = null;
    this._listeners = [];
  }

  onData(callback) {
    this._listeners.push(callback);
    return this;                       // chainable
  }

  start() {
    this.stop();
    const poll = async () => {
      try {
        const data = await this._fetchFn();
        this._listeners.forEach(cb => cb(data, null));
      } catch (err) {
        this._listeners.forEach(cb => cb(null, err));
      }
    };
    poll();                             // immediate first call
    this._timer = setInterval(poll, this._interval);
    return this;
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    return this;
  }

  setInterval(ms) {
    this._interval = ms;
    if (this._timer) this.start();     // restart with new interval
    return this;
  }
}

// ─── SSE (Server-Sent Events) Manager ─────────────────────────────────────────
class SSEClient {
  constructor(endpoint) {
    this._url = `${BASE_URL}${endpoint}`;
    this._source = null;
    this._handlers = {};
  }

  on(event, callback) {
    this._handlers[event] = callback;
    return this;
  }

  connect() {
    const token = TokenManager.accessToken;
    const url = token ? `${this._url}?token=${token}` : this._url;
    this._source = new EventSource(url);

    this._source.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (this._handlers['message']) this._handlers['message'](data);
    };

    Object.entries(this._handlers).forEach(([event, handler]) => {
      if (event !== 'message' && event !== 'error' && event !== 'open') {
        this._source.addEventListener(event, (e) => handler(JSON.parse(e.data)));
      }
    });

    this._source.onerror = (err) => {
      if (this._handlers['error']) this._handlers['error'](err);
      console.warn('SSE connection lost, reconnecting...');
    };

    this._source.onopen = () => {
      if (this._handlers['open']) this._handlers['open']();
    };

    return this;
  }

  close() {
    if (this._source) { this._source.close(); this._source = null; }
    return this;
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
window.BRX = {
  api: { request, healthCheck, BASE_URL, API_URL: API, ApiError },
  auth,
  vitals,
  steps,
  calories,
  predictions,
  anomaly,
  alerts,
  water,
  diet,
  whatsapp,
  datasets,
  devices,
  Poller,
  SSEClient,
};
